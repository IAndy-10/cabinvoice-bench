#pragma once

#include "../Common/FIFOAdapter.h"

struct DenoiseState;

/// Wrapper around the RNNoise C library.
/// RNNoise processes fixed 480-sample frames and is only valid at 48 kHz
/// (its band layout and model are fixed to that rate). At any other rate the
/// wrapper passes audio through untouched and reports zero latency.
/// A FIFOAdapter bridges variable host block sizes.
///
/// Latency (measured by Tests/TestLatency.cpp): FIFO (479) + RNNoise's own
/// analysis/synthesis overlap (480) = 959 samples = 19.98 ms at 48 kHz.
class RNNoiseWrapper {
public:
    static constexpr size_t kRNNoiseFrameSize = 480;
    static constexpr size_t kInternalDelay = 480;   // frame_synthesis overlap-add
    static constexpr double kRequiredSampleRate = 48000.0;
    static constexpr float kScale = 32768.0f;

    RNNoiseWrapper();
    ~RNNoiseWrapper();

    void prepare(double sampleRate);
    void process(float* data, size_t numSamples);
    bool isActive() const noexcept { return _active; }
    size_t getLatencySamples() const {
        return _active ? _fifo.getLatencySamples() + kInternalDelay : 0;
    }
    void reset();

private:
    void processRNNoiseFrame(float* input, float* output);

    FIFOAdapter _fifo;
    DenoiseState* _state = nullptr;
    bool _active = false;   // true when sr == 48 kHz and rnnoise_create succeeded
};
