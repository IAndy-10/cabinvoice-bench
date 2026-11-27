#pragma once

#include "../Common/SPSCRingBuffer.h"
#include "Downsampler.h"
#include "SileroModel.h"
#include <atomic>
#include <string>
#include <thread>

/// Real-time wrapper around SileroModel.
/// Audio thread: pushSamples() downsamples to 16 kHz into a lock-free SPSC ring.
/// Worker thread: reads 512-sample chunks, runs inference, publishes an atomic
/// probability and a hangover-stabilised speech flag.
class SileroVAD {
public:
    static constexpr size_t kRingSize = 16384;  // ~1 s at 16 kHz

    SileroVAD() : _ring(kRingSize) {}
    ~SileroVAD() { stop(); }

    /// Message thread. Loads the model (if a path is given) and starts the worker.
    /// Returns false with `error` set when the model could not be loaded; the
    /// worker still runs and reports "not speech", so the plugin keeps working.
    bool prepare(double hostSampleRate, const std::string& modelPath, std::string& error);

    /// Audio thread. Real-time safe.
    void pushSamples(const float* data, size_t numSamples) noexcept;

    bool  isSpeech() const noexcept       { return _isSpeech.load(std::memory_order_acquire); }
    float getProbability() const noexcept { return _probability.load(std::memory_order_acquire); }
    bool  isModelLoaded() const noexcept  { return _modelLoaded.load(std::memory_order_acquire); }

    /// Message thread. Stops and joins the worker.
    void stop();

private:
    void workerLoop();

    SPSCRingBuffer<float> _ring;
    Downsampler _downsampler;
    SileroModel _model;
    SileroModel::Hangover _hangover;   // worker thread only

    std::atomic<bool>  _isSpeech{ false };
    std::atomic<float> _probability{ 0.0f };
    std::atomic<bool>  _running{ false };
    std::atomic<bool>  _modelLoaded{ false };
    std::thread _workerThread;
};
