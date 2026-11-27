#pragma once

#include "../Common/SPSCRingBuffer.h"
#include <onnxruntime_cxx_api.h>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

/// Silero VAD running on a worker thread via ONNX Runtime.
/// Audio thread pushes 16 kHz downsampled audio into a lock-free SPSC ring.
/// Worker thread reads 512-sample chunks, runs inference, publishes an atomic speech flag.
class SileroVAD {
public:
    static constexpr size_t kChunkSize   = 512;   // 512 samples at 16 kHz = 32 ms
    static constexpr size_t kRingSize    = 16384;  // ~1 second buffer at 16 kHz
    static constexpr int    kHangoverMs  = 300;
    static constexpr float  kThreshold   = 0.5f;
    static constexpr int    kVadSampleRate = 16000;

    SileroVAD() : _ring(kRingSize) {}
    ~SileroVAD() { stop(); }

    void prepare(double hostSampleRate, const std::string& modelPath = "");

    /// Called from the audio thread. Push samples (will be downsampled to 16 kHz).
    void pushSamples(const float* data, size_t numSamples);

    bool isSpeech() const noexcept {
        return _isSpeech.load(std::memory_order_acquire);
    }

    float getProbability() const noexcept {
        return _probability.load(std::memory_order_acquire);
    }

    void stop();

private:
    void workerLoop();
    float runInference(const float* data, size_t numSamples);

    SPSCRingBuffer<float> _ring;
    std::atomic<bool> _isSpeech{false};
    std::atomic<float> _probability{0.0f};
    std::atomic<bool> _running{false};
    std::thread _workerThread;
    double _hostSampleRate = 48000.0;
    double _downsampleRatio = 3.0;
    std::atomic<int> _hangoverRemaining{0};

    // ONNX Runtime
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::MemoryInfo _memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Silero VAD state tensors (h, c for LSTM layers)
    std::vector<float> _hState;
    std::vector<float> _cState;
    int64_t _srState = kVadSampleRate;
    std::atomic<bool> _modelLoaded{false};
};
