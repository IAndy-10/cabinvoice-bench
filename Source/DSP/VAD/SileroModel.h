#pragma once

#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>

/// Synchronous Silero VAD v5 inference (ONNX Runtime). NOT real-time safe:
/// call it from a worker thread (plugin) or directly (CLI, tests).
///
/// v5 graph interface (checked at load time):
///   inputs : input  [1, 64 + 512] float  (64-sample context + 512-sample chunk at 16 kHz)
///            state  [2, 1, 128]   float
///            sr     []            int64  (scalar, 16000)
///   outputs: output [1, 1]        float  (speech probability)
///            stateN [2, 1, 128]   float
class SileroModel {
public:
    static constexpr int    kSampleRate  = 16000;
    static constexpr size_t kChunkSize   = 512;   // 32 ms at 16 kHz
    static constexpr size_t kContextSize = 64;
    static constexpr size_t kStateSize   = 2 * 1 * 128;

    /// Loads the model. Returns false and fills `error` on any failure
    /// (missing file, wrong interface, ORT error). Never throws.
    bool load(const std::string& modelPath, std::string& error);

    bool isLoaded() const noexcept { return _session != nullptr; }

    /// Clears recurrent state and context (start of a new stream).
    void reset() noexcept;

    /// Runs one 512-sample 16 kHz chunk; returns speech probability in [0, 1].
    /// Throws Ort::Exception on inference failure — callers must catch.
    float process(const float* chunk);

    /// Converts a speech-probability stream into a stable speech flag:
    /// onset at prob >= threshold, offset only after `hangoverMs` below it.
    struct Hangover {
        float threshold  = 0.5f;
        int   hangoverMs = 300;
        int   remainingMs = 0;
        bool  speech = false;

        /// Feed one 32 ms chunk's probability; returns the updated flag.
        bool update(float probability) noexcept {
            if (probability >= threshold) {
                speech = true;
                remainingMs = hangoverMs;
            } else if (remainingMs > 0) {
                remainingMs -= static_cast<int>(kChunkSize * 1000 / kSampleRate);
                if (remainingMs <= 0) speech = false;
            } else {
                speech = false;
            }
            return speech;
        }
        void reset() noexcept { remainingMs = 0; speech = false; }
    };

private:
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::MemoryInfo _memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<float, kContextSize + kChunkSize> _input{};
    std::array<float, kStateSize> _state{};
    int64_t _sr = kSampleRate;
};
