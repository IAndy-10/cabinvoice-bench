/// Offline CLI block processor for Clean Ears.
/// Reads a mono WAV, processes it through one of the 4 suppressor modes,
/// and writes the output WAV. Used by the Python evaluation harness.
///
/// Usage:
///   clean-ears-cli --mode <bypass|wiener|whisper-tuned|rnnoise>
///                  --input input.wav --output output.wav
///                  [--block-size 512] [--sample-rate 48000]

#include "../DSP/STFT/STFTEngine.h"
#include "../DSP/Wiener/WienerSuppressor.h"
#include "../DSP/RNNoise/RNNoiseWrapper.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

enum class Mode { Bypass, Wiener, RNNoise, WhisperTuned };

struct Args {
    Mode mode = Mode::Bypass;
    std::string inputPath;
    std::string outputPath;
    int blockSize = 512;
};

void printUsage(const char* progName)
{
    std::cerr << "Usage: " << progName
              << " --mode <bypass|wiener|rnnoise|whisper-tuned>"
              << " --input <wav> --output <wav>"
              << " [--block-size N]\n";
}

bool parseArgs(int argc, char* argv[], Args& args)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc)
        {
            std::string m = argv[++i];
            if      (m == "bypass")        args.mode = Mode::Bypass;
            else if (m == "wiener")        args.mode = Mode::Wiener;
            else if (m == "rnnoise")       args.mode = Mode::RNNoise;
            else if (m == "whisper-tuned") args.mode = Mode::WhisperTuned;
            else { std::cerr << "Unknown mode: " << m << "\n"; return false; }
        }
        else if (arg == "--input"  && i + 1 < argc) args.inputPath  = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.outputPath = argv[++i];
        else if (arg == "--block-size" && i + 1 < argc) args.blockSize = std::atoi(argv[++i]);
        else { std::cerr << "Unknown argument: " << arg << "\n"; return false; }
    }

    if (args.inputPath.empty() || args.outputPath.empty())
    {
        std::cerr << "Error: --input and --output are required.\n";
        return false;
    }
    if (args.blockSize <= 0 || args.blockSize > 65536)
    {
        std::cerr << "Error: --block-size must be in [1, 65536].\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    // Initialise JUCE for non-GUI use
    juce::ScopedJuceInitialiser_GUI juceInit;

    Args args;
    if (!parseArgs(argc, argv, args))
    {
        printUsage(argv[0]);
        return 1;
    }

    // ── Read input WAV ──────────────────────────────────────────
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    juce::File inputFile(args.inputPath);
    if (!inputFile.existsAsFile())
    {
        std::cerr << "Error: input file not found: " << args.inputPath << "\n";
        return 1;
    }

    std::unique_ptr<juce::AudioFormatReader> reader(
        formatManager.createReaderFor(inputFile));

    if (reader == nullptr)
    {
        std::cerr << "Error: could not read audio file: " << args.inputPath << "\n";
        return 1;
    }

    const auto numSamples   = static_cast<int>(reader->lengthInSamples);
    const auto sampleRate   = reader->sampleRate;
    const auto numChannels  = static_cast<int>(reader->numChannels);

    // Read all samples into a mono buffer
    juce::AudioBuffer<float> buffer(1, numSamples);
    if (numChannels == 1)
    {
        reader->read(&buffer, 0, numSamples, 0, true, false);
    }
    else
    {
        // Mix to mono
        juce::AudioBuffer<float> multiCh(numChannels, numSamples);
        reader->read(&multiCh, 0, numSamples, 0, true, true);
        buffer.clear();
        for (int ch = 0; ch < numChannels; ++ch)
            buffer.addFrom(0, 0, multiCh, ch, 0, numSamples, 1.0f / numChannels);
    }

    // ── Prepare DSP ─────────────────────────────────────────────
    STFTEngine stft;
    WienerSuppressor wiener;
    RNNoiseWrapper rnnoise;

    stft.prepare(sampleRate);
    rnnoise.prepare(sampleRate);

    WienerSuppressor::Config wienerConfig;
    switch (args.mode)
    {
        case Mode::Wiener:
            wienerConfig = { 0.98f, 0.05f };
            break;
        case Mode::WhisperTuned:
            wienerConfig = { 0.92f, 0.10f };
            break;
        default:
            break;
    }

    if (args.mode == Mode::Wiener || args.mode == Mode::WhisperTuned)
        wiener.prepare(STFTEngine::kNumBins, wienerConfig);

    // ── Process in blocks ───────────────────────────────────────
    // For offline processing, VAD is not available. The Wiener suppressor
    // treats the first ~500 ms as silence to build the noise estimate,
    // then switches to speech mode. This matches a simple "noise-first"
    // assumption common in offline evaluation.
    
    const int noisePrefixSamples = static_cast<int>(0.5 * sampleRate);
    auto* data = buffer.getWritePointer(0);
    const int blockSize = args.blockSize;

    for (int pos = 0; pos < numSamples; pos += blockSize)
    {
        const int thisBlock = std::min(blockSize, numSamples - pos);
        const bool isSpeech = (pos + thisBlock) > noisePrefixSamples;

        switch (args.mode)
        {
            case Mode::Bypass:
                break;

            case Mode::Wiener:
            case Mode::WhisperTuned:
                stft.process(data + pos, static_cast<size_t>(thisBlock),
                    [&](std::complex<float>* bins, size_t numBins) {
                        wiener.process(bins, numBins, isSpeech);
                    });
                break;

            case Mode::RNNoise:
                rnnoise.process(data + pos, static_cast<size_t>(thisBlock));
                break;
        }
    }

    // ── Write output WAV ────────────────────────────────────────
    juce::File outputFile(args.outputPath);
    outputFile.getParentDirectory().createDirectory();

    std::unique_ptr<juce::FileOutputStream> outStream(outputFile.createOutputStream());
    if (outStream == nullptr)
    {
        std::cerr << "Error: could not create output file: " << args.outputPath << "\n";
        return 1;
    }

    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        wavFormat.createWriterFor(outStream.get(),
                                  sampleRate,
                                  1,           // mono
                                  16,          // bits
                                  {},          // metadata
                                  0));         // quality

    if (writer == nullptr)
    {
        std::cerr << "Error: could not create WAV writer.\n";
        return 1;
    }

    outStream.release(); // writer takes ownership

    writer->writeFromAudioSampleBuffer(buffer, 0, numSamples);
    writer.reset(); // flush

    std::cout << "Processed " << numSamples << " samples ("
              << static_cast<double>(numSamples) / sampleRate << " s) "
              << "at " << sampleRate << " Hz, mode="
              << (args.mode == Mode::Bypass       ? "bypass" :
                  args.mode == Mode::Wiener        ? "wiener" :
                  args.mode == Mode::RNNoise       ? "rnnoise" :
                                                     "whisper-tuned")
              << "\n";

    return 0;
}
