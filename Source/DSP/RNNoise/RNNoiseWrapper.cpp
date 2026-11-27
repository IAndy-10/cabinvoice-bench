#include "RNNoiseWrapper.h"
#include <rnnoise.h>

RNNoiseWrapper::RNNoiseWrapper() = default;

RNNoiseWrapper::~RNNoiseWrapper()
{
    if (_state != nullptr)
        rnnoise_destroy(_state);
}

void RNNoiseWrapper::prepare(double sampleRate)
{
    if (_state != nullptr)
        rnnoise_destroy(_state);
    _state = nullptr;

    if (sampleRate == kRequiredSampleRate)
        _state = rnnoise_create(nullptr);   // may return null on allocation failure

    _active = (_state != nullptr);
    _fifo.prepare(kRNNoiseFrameSize);
}

void RNNoiseWrapper::process(float* data, size_t numSamples)
{
    if (!_active)
        return;   // pass-through, zero latency

    _fifo.process(data, data, numSamples,
        [this](float* frameIn, float* frameOut) {
            processRNNoiseFrame(frameIn, frameOut);
        });
}

void RNNoiseWrapper::reset()
{
    _fifo.reset();
    // Do NOT destroy/recreate _state here — rnnoise_create mallocs.
    // This is called from processBlock on mode switch.
    // RNNoise internal state will naturally adapt within a few frames.
}

void RNNoiseWrapper::processRNNoiseFrame(float* input, float* output)
{
    // RNNoise expects samples in [-32768, 32768] range
    for (size_t i = 0; i < kRNNoiseFrameSize; ++i)
        input[i] *= kScale;

    rnnoise_process_frame(_state, output, input);

    // Scale back to [-1, 1] range
    for (size_t i = 0; i < kRNNoiseFrameSize; ++i)
        output[i] /= kScale;
}
