/// Mode latency test.
/// Runs a voiced, speech-like test signal through each processing mode with
/// irregular host block sizes, measures the input->output delay by
/// cross-correlation, and checks it equals the latency the mode reports.
/// Prints the measured values so the README latency column comes from here.
/// Returns 0 on pass, 1 on fail.

#include "../Source/DSP/STFT/STFTEngine.h"
#include "../Source/DSP/Wiener/WienerSuppressor.h"
#include "../Source/DSP/RNNoise/RNNoiseWrapper.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double kFs = 48000.0;
constexpr size_t kLen = 48000 * 3;

/// 120 Hz glottal-like pulse train through a crude two-formant resonator,
/// syllable-rate amplitude modulation, plus a little noise. RNNoise treats it
/// as voice and keeps it, which an impulse or white noise would not survive.
std::vector<float> makeVoicedSignal()
{
    std::vector<float> x(kLen, 0.0f);
    std::mt19937 rng(1234);
    std::normal_distribution<float> noise(0.0f, 0.002f);

    const int period = static_cast<int>(kFs / 120.0);
    double y1 = 0, y2 = 0, z1 = 0, z2 = 0;
    auto resonator = [](double f, double bw, double& a1, double& a2) {
        const double r = std::exp(-M_PI * bw / kFs);
        a1 = 2.0 * r * std::cos(2.0 * M_PI * f / kFs);
        a2 = -r * r;
    };
    double a1, a2, c1, c2;
    resonator(700.0, 120.0, a1, a2);
    resonator(1200.0, 150.0, c1, c2);

    for (size_t n = 0; n < kLen; ++n) {
        const double pulse = (n % period == 0) ? 1.0 : 0.0;
        const double f1 = pulse + a1 * y1 + a2 * y2;  y2 = y1; y1 = f1;
        const double f2 = f1    + c1 * z1 + c2 * z2;  z2 = z1; z1 = f2;
        const double env = 0.5 * (1.0 - std::cos(2.0 * M_PI * 4.0 * n / kFs)); // 4 Hz syllables
        x[n] = static_cast<float>(0.02 * f2 * env) + noise(rng);
    }
    return x;
}

template <typename ProcessFn>
std::vector<float> runBlocks(std::vector<float> data, ProcessFn&& fn)
{
    const size_t blocks[] = { 64, 128, 256, 512, 1024, 480, 333, 17 };
    size_t pos = 0, i = 0;
    while (pos < data.size()) {
        size_t n = std::min(blocks[i++ % 8], data.size() - pos);
        fn(data.data() + pos, n);
        pos += n;
    }
    return data;
}

/// Lag (in samples, 0..maxLag) maximising normalised cross-correlation,
/// measured on the second half of the signal so start-up transients don't count.
long measureDelay(const std::vector<float>& in, const std::vector<float>& out, long maxLag)
{
    const size_t start = kLen / 2, end = kLen - static_cast<size_t>(maxLag) - 1;
    double best = -1.0;
    long bestLag = -1;
    for (long lag = 0; lag <= maxLag; ++lag) {
        double xy = 0, xx = 0, yy = 0;
        for (size_t n = start; n < end; ++n) {
            const double a = in[n], b = out[n + static_cast<size_t>(lag)];
            xy += a * b; xx += a * a; yy += b * b;
        }
        const double r = xy / std::sqrt(xx * yy + 1e-30);
        if (r > best) { best = r; bestLag = lag; }
    }
    return bestLag;
}

bool check(const char* name, long measured, size_t reported)
{
    const bool ok = measured == static_cast<long>(reported);
    std::printf("  %-10s reported %5zu  measured %5ld  (%.2f ms)  %s\n",
                name, reported, measured, 1000.0 * measured / kFs, ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main()
{
    const auto input = makeVoicedSignal();
    bool ok = true;

    std::printf("Mode latency test (48 kHz, irregular blocks)\n");

    {   // STFT identity path
        STFTEngine stft; stft.prepare(kFs);
        auto out = runBlocks(input, [&](float* d, size_t n) {
            stft.process(d, n, [](std::complex<float>*, size_t) {});
        });
        ok &= check("stft", measureDelay(input, out, 3000), stft.getLatencySamples());
    }

    {   // Wiener (noise estimate never updated -> gains near 1, phase untouched)
        STFTEngine stft; stft.prepare(kFs);
        WienerSuppressor w; w.prepare(STFTEngine::kNumBins, {});
        auto out = runBlocks(input, [&](float* d, size_t n) {
            stft.process(d, n, [&](std::complex<float>* b, size_t k) { w.process(b, k, true); });
        });
        ok &= check("wiener", measureDelay(input, out, 3000), stft.getLatencySamples());
    }

    {   // RNNoise
        RNNoiseWrapper rn; rn.prepare(kFs);
        auto out = runBlocks(input, [&](float* d, size_t n) { rn.process(d, n); });
        ok &= check("rnnoise", measureDelay(input, out, 3000), rn.getLatencySamples());
    }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
