/// Silero VAD v5 + side-chain downsampler test.
///   1. The shipped model loads and exposes the v5 interface.
///   2. Inference runs with state carried across chunks (no exceptions).
///   3. Silence and white noise stay below the speech threshold.
///   4. The 48k -> 16k downsampler passes in-band tones and rejects tones
///      that would alias into the VAD band.
/// Usage: TestSileroVAD <path/to/silero_vad.onnx>. Returns 0 on pass, 1 on fail.

#include "../Source/DSP/VAD/Downsampler.h"
#include "../Source/DSP/VAD/SileroModel.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

float maxProbability(SileroModel& m, const std::vector<float>& x16k)
{
    m.reset();
    float maxP = 0.0f;
    for (size_t pos = 0; pos + SileroModel::kChunkSize <= x16k.size(); pos += SileroModel::kChunkSize)
        maxP = std::max(maxP, m.process(x16k.data() + pos));
    return maxP;
}

double toneGainDb(double freq)
{
    constexpr double fs = 48000.0;
    Downsampler ds; ds.prepare(fs);
    std::vector<float> in(48000);
    for (size_t n = 0; n < in.size(); ++n)
        in[n] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * freq * n / fs));

    std::vector<float> out;
    // Irregular blocks: phase must carry across block boundaries.
    const size_t blocks[] = { 480, 512, 333, 1024, 17 };
    for (size_t pos = 0, i = 0; pos < in.size(); ) {
        const size_t n = std::min(blocks[i++ % 5], in.size() - pos);
        ds.process(in.data() + pos, n, [&](float s) { out.push_back(s); });
        pos += n;
    }

    double sumSq = 0.0; size_t count = 0;
    for (size_t n = out.size() / 4; n < out.size(); ++n) { sumSq += out[n] * out[n]; ++count; }
    const double rms = std::sqrt(sumSq / count);
    return 20.0 * std::log10(rms / (0.5 / std::sqrt(2.0)));
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) { std::printf("usage: %s <silero_vad.onnx>\n", argv[0]); return 1; }
    bool ok = true;

    std::printf("Silero VAD v5 test\n");

    SileroModel model;
    std::string error;
    if (!model.load(argv[1], error)) {
        std::printf("  load: FAIL (%s)\n", error.c_str());
        return 1;
    }
    std::printf("  load + v5 interface + warm-up: PASS\n");

    try {
        const std::vector<float> silence(16000 * 2, 0.0f);
        std::vector<float> noise(16000 * 2);
        std::mt19937 rng(7);
        std::normal_distribution<float> g(0.0f, 0.05f);
        for (auto& s : noise) s = g(rng);

        const float pSil = maxProbability(model, silence);
        const float pNoise = maxProbability(model, noise);
        const bool okSil = pSil < 0.5f, okNoise = pNoise < 0.5f;
        std::printf("  silence max p = %.3f  %s\n", pSil, okSil ? "PASS" : "FAIL");
        std::printf("  white noise max p = %.3f  %s\n", pNoise, okNoise ? "PASS" : "FAIL");
        ok &= okSil && okNoise;
    } catch (const Ort::Exception& e) {
        std::printf("  inference: FAIL (%s)\n", e.what());
        return 1;
    }

    const double pass1k = toneGainDb(1000.0);
    const double stop12k = toneGainDb(12000.0);  // would alias to 4 kHz with plain decimation
    const bool okPass = std::abs(pass1k) < 1.0, okStop = stop12k < -30.0;
    std::printf("  downsampler 1 kHz gain = %+.2f dB  %s\n", pass1k, okPass ? "PASS" : "FAIL");
    std::printf("  downsampler 12 kHz alias = %+.2f dB  %s\n", stop12k, okStop ? "PASS" : "FAIL");
    ok &= okPass && okStop;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
