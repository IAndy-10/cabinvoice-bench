#include "SileroVAD.h"
#include <array>
#include <chrono>

bool SileroVAD::prepare(double hostSampleRate, const std::string& modelPath, std::string& error)
{
    stop();

    _downsampler.prepare(hostSampleRate, SileroModel::kSampleRate);
    _ring.reset();
    _hangover.reset();
    _isSpeech.store(false, std::memory_order_relaxed);
    _probability.store(0.0f, std::memory_order_relaxed);

    bool loaded = false;
    if (modelPath.empty())
        error = "no model path given";
    else
        loaded = _model.load(modelPath, error);

    _modelLoaded.store(loaded, std::memory_order_release);
    _running.store(true, std::memory_order_release);
    _workerThread = std::thread([this] { workerLoop(); });
    return loaded;
}

void SileroVAD::stop()
{
    _running.store(false, std::memory_order_release);
    if (_workerThread.joinable())
        _workerThread.join();
}

void SileroVAD::pushSamples(const float* data, size_t numSamples) noexcept
{
    // Batch the 16 kHz output to keep ring writes (and their atomics) cheap.
    std::array<float, 256> batch;
    size_t count = 0;

    _downsampler.process(data, numSamples, [&](float s) {
        batch[count++] = s;
        if (count == batch.size()) {
            _ring.write(batch.data(), count);   // drops samples if the worker stalls
            count = 0;
        }
    });

    if (count > 0)
        _ring.write(batch.data(), count);
}

void SileroVAD::workerLoop()
{
    std::array<float, SileroModel::kChunkSize> chunk{};

    while (_running.load(std::memory_order_acquire))
    {
        if (_ring.availableRead() < chunk.size())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        _ring.read(chunk.data(), chunk.size());

        if (!_modelLoaded.load(std::memory_order_acquire))
            continue;   // drain the ring; flag stays false

        try
        {
            const float p = _model.process(chunk.data());
            _probability.store(p, std::memory_order_release);
            _isSpeech.store(_hangover.update(p), std::memory_order_release);
        }
        catch (const Ort::Exception&)
        {
            // Never let an exception escape a std::thread (std::terminate would
            // take the host down). Disable the VAD and report not-speech.
            _modelLoaded.store(false, std::memory_order_release);
            _isSpeech.store(false, std::memory_order_release);
            _probability.store(0.0f, std::memory_order_release);
        }
    }
}
