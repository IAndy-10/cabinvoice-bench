#pragma once

#include "../Common/FIFOAdapter.h"

struct DenoiseState;

/// Wrapper around the RNNoise C library.
/// RNNoise processes fixed 480-sample frames at 48 kHz (10 ms).
/// This wrapper uses a FIFOAdapter to bridge variable host block sizes.
class RNNoiseWrapper {
public:
    static constexpr size_t kRNNoiseFrameSize = 480;
    static constexpr float kScale = 32768.0f;

    RNNoiseWrapper();
    ~RNNoiseWrapper();

    void prepare(double sampleRate);
    void process(float* data, size_t numSamples);
    size_t getLatencySamples() const { return _fifo.getLatencySamples(); }
    void reset();

private:
    void processRNNoiseFrame(float* input, float* output);

    FIFOAdapter _fifo;
    DenoiseState* _state = nullptr;
};
