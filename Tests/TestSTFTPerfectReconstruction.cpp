/// STFT Perfect Reconstruction Test
/// Feeds a sine wave through the STFT engine with an identity callback (all gains = 1)
/// and verifies the output matches the input within floating-point tolerance.
/// This is a standalone C++ test — no test framework needed. Returns 0 on pass, 1 on fail.

#include "../Source/DSP/STFT/STFTEngine.h"
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr double kSampleRate = 48000.0;
static constexpr size_t kTestLengthSamples = 48000; // 1 second
static constexpr float  kFreqHz = 440.0f;
static constexpr float  kTolerance = 1e-4f;

int main()
{
    STFTEngine stft;
    stft.prepare(kSampleRate);

    const size_t latency = stft.getLatencySamples();

    // Generate sine wave
    std::vector<float> input(kTestLengthSamples);
    for (size_t i = 0; i < kTestLengthSamples; ++i)
        input[i] = std::sin(2.0f * static_cast<float>(M_PI) * kFreqHz
                            * static_cast<float>(i) / static_cast<float>(kSampleRate));

    // Process through STFT with identity callback (no spectral modification)
    std::vector<float> output = input;

    // Process in variable block sizes to simulate real host behavior
    const size_t blockSizes[] = {64, 128, 256, 512, 1024, 480, 333};
    size_t pos = 0;
    size_t blockIdx = 0;

    while (pos < kTestLengthSamples) {
        size_t blockSize = blockSizes[blockIdx % 7];
        if (pos + blockSize > kTestLengthSamples)
            blockSize = kTestLengthSamples - pos;

        stft.process(output.data() + pos, blockSize,
            [](std::complex<float>* /*bins*/, size_t /*numBins*/) {
                // Identity: leave all bins untouched
            });

        pos += blockSize;
        ++blockIdx;
    }

    // Compare input vs output, skipping the latency region
    float maxError = 0.0f;
    size_t errorCount = 0;
    const size_t compareStart = latency + STFTEngine::kFrameSize; // skip transient
    const size_t compareEnd   = kTestLengthSamples - STFTEngine::kFrameSize;

    for (size_t i = compareStart; i < compareEnd; ++i) {
        float expected = input[i - latency];
        float actual   = output[i];
        float err = std::abs(expected - actual);

        if (err > maxError)
            maxError = err;

        if (err > kTolerance) {
            if (errorCount < 5)
                std::printf("  MISMATCH at sample %zu: expected %.6f, got %.6f (err=%.6f)\n",
                            i, expected, actual, err);
            ++errorCount;
        }
    }

    std::printf("STFT Perfect Reconstruction Test\n");
    std::printf("  Latency: %zu samples (%.1f ms)\n", latency, 1000.0 * latency / kSampleRate);
    std::printf("  Compared samples: %zu - %zu\n", compareStart, compareEnd);
    std::printf("  Max error: %.8f\n", maxError);
    std::printf("  Errors (>%.1e): %zu\n", kTolerance, errorCount);

    if (errorCount == 0) {
        std::printf("  PASS\n");
        return 0;
    } else {
        std::printf("  FAIL\n");
        return 1;
    }
}
