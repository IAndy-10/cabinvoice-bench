/// Offline CLI block processor for CabinVoice Bench.
///
/// Runs a 48 kHz WAV through the same DSP objects the plugin uses, in host-sized
/// blocks, and writes a latency-aligned output WAV (sample n of the output
/// corresponds to sample n of the input, same length, for every mode).
///
/// Noise-floor updates are gated by the same Silero VAD model, threshold and
/// hangover as the plugin. Offline, the VAD flag for a block is the decision of
/// the last 32 ms chunk that ends inside or before that block, i.e. the plugin's
/// behaviour with zero inference delay.
///
/// Usage:
///   cabinvoice-bench-cli --mode <bypass|wiener|rnnoise|whisper-tuned>
///                  --input in.wav --output out.wav
///                  [--strength 0..1]        Wiener applied gain = G^strength (default 1)
///                  [--vad-model path.onnx]  default: models/silero_vad.onnx
///                  [--no-vad]               legacy: first 500 ms = noise, then frozen
///                  [--vad-events out.json]  VAD probabilities + segments for the
///                                           input and the processed output
///                  [--block-size 512]
///
/// Input must be 48 kHz (RNNoise is only valid there, and all modes are run at
/// the same rate for a fair comparison). Resample upstream with a zero-phase
/// resampler; the eval harness does this.

#include "../DSP/STFT/STFTEngine.h"
#include "../DSP/Wiener/WienerSuppressor.h"
#include "../DSP/RNNoise/RNNoiseWrapper.h"
#include "../DSP/VAD/Downsampler.h"
#include "../DSP/VAD/SileroModel.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum class Mode { Bypass, Wiener, RNNoise, WhisperTuned };

constexpr double kRequiredRate = 48000.0;

struct Args {
    Mode mode = Mode::Bypass;
    std::string modeName = "bypass";
    std::string inputPath;
    std::string outputPath;
    std::string vadModelPath = "models/silero_vad.onnx";
    std::string vadEventsPath;
    float strength = 1.0f;
    bool useVad = true;
    int blockSize = 512;
};

void printUsage(const char* progName)
{
    std::cerr << "Usage: " << progName
              << " --mode <bypass|wiener|rnnoise|whisper-tuned> --input <wav> --output <wav>\n"
                 "       [--strength 0..1] [--vad-model path] [--no-vad]\n"
                 "       [--vad-events path.json] [--block-size N]\n";
}

bool parseArgs(int argc, char* argv[], Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };

        if (arg == "--mode") {
            const char* m = next();
            if (m == nullptr) return false;
            args.modeName = m;
            if      (args.modeName == "bypass")        args.mode = Mode::Bypass;
            else if (args.modeName == "wiener")        args.mode = Mode::Wiener;
            else if (args.modeName == "rnnoise")       args.mode = Mode::RNNoise;
            else if (args.modeName == "whisper-tuned") args.mode = Mode::WhisperTuned;
            else { std::cerr << "Unknown mode: " << m << "\n"; return false; }
        }
        else if (arg == "--input")      { auto v = next(); if (!v) return false; args.inputPath = v; }
        else if (arg == "--output")     { auto v = next(); if (!v) return false; args.outputPath = v; }
        else if (arg == "--vad-model")  { auto v = next(); if (!v) return false; args.vadModelPath = v; }
        else if (arg == "--vad-events") { auto v = next(); if (!v) return false; args.vadEventsPath = v; }
        else if (arg == "--strength")   { auto v = next(); if (!v) return false; args.strength = std::strtof(v, nullptr); }
        else if (arg == "--block-size") { auto v = next(); if (!v) return false; args.blockSize = std::atoi(v); }
        else if (arg == "--no-vad")     { args.useVad = false; }
        else { std::cerr << "Unknown argument: " << arg << "\n"; return false; }
    }

    if (args.inputPath.empty() || args.outputPath.empty()) {
        std::cerr << "Error: --input and --output are required.\n";
        return false;
    }
    if (args.blockSize <= 0 || args.blockSize > 65536) {
        std::cerr << "Error: --block-size must be in [1, 65536].\n";
        return false;
    }
    if (!(args.strength >= 0.0f && args.strength <= 1.0f)) {
        std::cerr << "Error: --strength must be in [0, 1].\n";
        return false;
    }
    return true;
}

/// Per-32 ms-chunk VAD track over a whole 48 kHz signal.
struct VadTrack {
    std::vector<float> prob;      // one per chunk
    std::vector<bool>  speech;    // hangover-stabilised flag per chunk
    double samplesPerChunk = 0;   // in 48 kHz samples

    /// Flag in force at 48 kHz sample position `pos` (last chunk ending at or before it).
    bool flagAt(size_t pos) const {
        if (speech.empty()) return false;
        const auto done = static_cast<long>(static_cast<double>(pos) / samplesPerChunk) - 1;
        if (done < 0) return false;
        return speech[std::min(static_cast<size_t>(done), speech.size() - 1)];
    }
};

VadTrack runVad(SileroModel& model, const float* x, size_t n)
{
    VadTrack t;
    t.samplesPerChunk = SileroModel::kChunkSize * kRequiredRate / SileroModel::kSampleRate;

    Downsampler ds;
    ds.prepare(kRequiredRate, SileroModel::kSampleRate);
    model.reset();
    SileroModel::Hangover hang;

    std::vector<float> chunk;
    chunk.reserve(SileroModel::kChunkSize);
    ds.process(x, n, [&](float s) {
        chunk.push_back(s);
        if (chunk.size() == SileroModel::kChunkSize) {
            const float p = model.process(chunk.data());
            t.prob.push_back(p);
            t.speech.push_back(hang.update(p));
            chunk.clear();
        }
    });
    return t;
}

juce::var trackToJson(const VadTrack& t)
{
    juce::Array<juce::var> probs, segments;
    for (float p : t.prob)
        probs.add(std::round(p * 1000.0f) / 1000.0f);

    const double chunkSec = t.samplesPerChunk / kRequiredRate;
    for (size_t i = 0; i < t.speech.size(); ) {
        if (!t.speech[i]) { ++i; continue; }
        size_t j = i;
        while (j < t.speech.size() && t.speech[j]) ++j;
        juce::Array<juce::var> seg;
        seg.add(i * chunkSec);   // onset: start of first speech chunk
        seg.add(j * chunkSec);   // offset: end of last flagged chunk (includes hangover)
        segments.add(seg);
        i = j;
    }

    auto* obj = new juce::DynamicObject();
    obj->setProperty("prob", probs);
    obj->setProperty("segments", segments);
    obj->setProperty("num_onsets", segments.size());
    return juce::var(obj);
}

} // namespace

int main(int argc, char* argv[])
{
    Args args;
    if (!parseArgs(argc, argv, args)) { printUsage(argv[0]); return 1; }

    // ── Read input ──────────────────────────────────────────────
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    const juce::File inputFile(args.inputPath);
    std::unique_ptr<juce::AudioFormatReader> reader(
        inputFile.existsAsFile() ? formatManager.createReaderFor(inputFile) : nullptr);
    if (reader == nullptr) {
        std::cerr << "Error: could not read audio file: " << args.inputPath << "\n";
        return 1;
    }
    if (reader->sampleRate != kRequiredRate) {
        std::cerr << "Error: input is " << reader->sampleRate << " Hz; cabinvoice-bench-cli needs "
                  << kRequiredRate << " Hz. Resample upstream (zero-phase), e.g. "
                     "scipy.signal.resample_poly.\n";
        return 1;
    }

    const auto numSamples  = static_cast<int>(reader->lengthInSamples);
    const auto numChannels = static_cast<int>(reader->numChannels);

    juce::AudioBuffer<float> input(1, numSamples);
    {
        juce::AudioBuffer<float> multi(numChannels, numSamples);
        reader->read(&multi, 0, numSamples, 0, true, numChannels > 1);
        input.clear();
        for (int ch = 0; ch < numChannels; ++ch)
            input.addFrom(0, 0, multi, ch, 0, numSamples, 1.0f / static_cast<float>(numChannels));
    }

    // ── DSP setup (same objects and parameters as the plugin) ───
    STFTEngine stft;
    WienerSuppressor wiener;
    RNNoiseWrapper rnnoise;
    stft.prepare(kRequiredRate);
    rnnoise.prepare(kRequiredRate);

    WienerSuppressor::Config cfg { 0.98f, 0.05f, args.strength };
    if (args.mode == Mode::WhisperTuned)
        cfg = { 0.92f, 0.10f, args.strength };   // legacy preset kept for old result files
    wiener.prepare(STFTEngine::kNumBins, cfg);

    size_t latency = 0;
    switch (args.mode) {
        case Mode::Bypass:       latency = 0; break;
        case Mode::Wiener:
        case Mode::WhisperTuned: latency = stft.getLatencySamples(); break;
        case Mode::RNNoise:      latency = rnnoise.getLatencySamples(); break;
    }

    // ── VAD on the input (gates the noise floor, like the plugin) ─
    SileroModel vadModel;
    VadTrack inputVad;
    const bool needVad = args.useVad
        && (args.mode == Mode::Wiener || args.mode == Mode::WhisperTuned || !args.vadEventsPath.empty());
    if (needVad) {
        std::string err;
        if (!vadModel.load(args.vadModelPath, err)) {
            std::cerr << "Error: could not load VAD model '" << args.vadModelPath << "': " << err
                      << "\n       (pass --vad-model, or --no-vad for the legacy noise-prefix mode)\n";
            return 1;
        }
        try { inputVad = runVad(vadModel, input.getReadPointer(0), static_cast<size_t>(numSamples)); }
        catch (const Ort::Exception& e) { std::cerr << "Error: VAD inference failed: " << e.what() << "\n"; return 1; }
    }

    // ── Process with latency compensation ───────────────────────
    // Pad by `latency` zeros so the tail is flushed, process in host-sized
    // blocks, then drop the first `latency` samples.
    const size_t padded = static_cast<size_t>(numSamples) + latency;
    std::vector<float> work(padded, 0.0f);
    std::copy(input.getReadPointer(0), input.getReadPointer(0) + numSamples, work.begin());

    const size_t noisePrefix = static_cast<size_t>(0.5 * kRequiredRate);
    const size_t block = static_cast<size_t>(args.blockSize);

    for (size_t pos = 0; pos < padded; pos += block) {
        const size_t n = std::min(block, padded - pos);
        const bool isSpeech = args.useVad ? inputVad.flagAt(pos + n) : (pos + n > noisePrefix);
        float* d = work.data() + pos;

        switch (args.mode) {
            case Mode::Bypass: break;
            case Mode::Wiener:
            case Mode::WhisperTuned:
                stft.process(d, n, [&](std::complex<float>* bins, size_t numBins) {
                    wiener.process(bins, numBins, isSpeech);
                });
                break;
            case Mode::RNNoise:
                rnnoise.process(d, n);
                break;
        }
    }

    juce::AudioBuffer<float> output(1, numSamples);
    std::copy(work.begin() + static_cast<long>(latency),
              work.begin() + static_cast<long>(latency) + numSamples,
              output.getWritePointer(0));

    // ── Write output (24-bit, 48 kHz, mono) ─────────────────────
    const juce::File outputFile(args.outputPath);
    outputFile.getParentDirectory().createDirectory();
    outputFile.deleteFile();
    std::unique_ptr<juce::OutputStream> outStream(outputFile.createOutputStream().release());
    if (outStream == nullptr) {
        std::cerr << "Error: could not create output file: " << args.outputPath << "\n";
        return 1;
    }
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wav.createWriterFor(outStream.get(), kRequiredRate, 1, 24, {}, 0));
    if (writer == nullptr) {
        std::cerr << "Error: could not create WAV writer.\n";
        return 1;
    }
    outStream.release();   // writer owns the stream now
    writer->writeFromAudioSampleBuffer(output, 0, numSamples);
    writer.reset();

    // ── VAD events JSON ─────────────────────────────────────────
    if (!args.vadEventsPath.empty() && args.useVad) {
        VadTrack outputVad;
        try { outputVad = runVad(vadModel, output.getReadPointer(0), static_cast<size_t>(numSamples)); }
        catch (const Ort::Exception& e) { std::cerr << "Error: VAD inference failed: " << e.what() << "\n"; return 1; }

        auto* root = new juce::DynamicObject();
        root->setProperty("mode", juce::String(args.modeName));
        root->setProperty("strength", args.strength);
        root->setProperty("sample_rate", kRequiredRate);
        root->setProperty("latency_samples", static_cast<int>(latency));
        root->setProperty("chunk_ms", 1000.0 * SileroModel::kChunkSize / SileroModel::kSampleRate);
        root->setProperty("threshold", SileroModel::Hangover{}.threshold);
        root->setProperty("hangover_ms", SileroModel::Hangover{}.hangoverMs);
        root->setProperty("input", trackToJson(inputVad));
        root->setProperty("output", trackToJson(outputVad));

        const juce::File jsonFile(args.vadEventsPath);
        jsonFile.getParentDirectory().createDirectory();
        if (!jsonFile.replaceWithText(juce::JSON::toString(juce::var(root), true))) {
            std::cerr << "Error: could not write " << args.vadEventsPath << "\n";
            return 1;
        }
    }

    std::cout << "Processed " << numSamples << " samples ("
              << numSamples / kRequiredRate << " s), mode=" << args.modeName
              << ", strength=" << args.strength
              << ", latency compensated=" << latency << " samples"
              << ", vad=" << (args.useVad ? "silero" : "noise-prefix") << "\n";
    return 0;
}
