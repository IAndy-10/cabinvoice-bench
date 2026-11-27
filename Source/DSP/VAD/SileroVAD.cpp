#include "SileroVAD.h"
#include <chrono>
#include <cstring>

void SileroVAD::prepare(double hostSampleRate, const std::string& modelPath)
{
    stop();

    _hostSampleRate = hostSampleRate;
    _downsampleRatio = hostSampleRate / static_cast<double>(kVadSampleRate);
    _ring.reset();
    _isSpeech.store(false, std::memory_order_relaxed);
    _probability.store(0.0f, std::memory_order_relaxed);
    _hangoverRemaining.store(0, std::memory_order_relaxed);
    _modelLoaded.store(false, std::memory_order_relaxed);

    if (!modelPath.empty())
    {
        try
        {
            _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "vad");

            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            _session = std::make_unique<Ort::Session>(*_env, modelPath.c_str(), opts);

            // Silero VAD v5 LSTM state: 2 layers, batch 1, 64 hidden units
            _hState.assign(2 * 1 * 64, 0.0f);
            _cState.assign(2 * 1 * 64, 0.0f);
            _srState = kVadSampleRate;

            // Warm-up inference
            std::vector<float> dummy(kChunkSize, 0.0f);
            runInference(dummy.data(), kChunkSize);

            // Reset state after warmup
            std::fill(_hState.begin(), _hState.end(), 0.0f);
            std::fill(_cState.begin(), _cState.end(), 0.0f);

            _modelLoaded.store(true, std::memory_order_release);
        }
        catch (const Ort::Exception&)
        {
            _session.reset();
            _env.reset();
            _modelLoaded.store(false, std::memory_order_release);
        }
    }

    _running.store(true, std::memory_order_release);
    _workerThread = std::thread([this] { workerLoop(); });
}

void SileroVAD::stop()
{
    _running.store(false, std::memory_order_release);
    if (_workerThread.joinable())
        _workerThread.join();
    _session.reset();
    _env.reset();
    _modelLoaded.store(false, std::memory_order_relaxed);
}

void SileroVAD::pushSamples(const float* data, size_t numSamples)
{
    // Simple decimation from host rate to 16 kHz
    for (double pos = 0.0; pos < static_cast<double>(numSamples); pos += _downsampleRatio)
    {
        auto idx = static_cast<size_t>(pos);
        if (idx < numSamples)
        {
            float sample = data[idx];
            _ring.write(&sample, 1);
        }
    }
}

void SileroVAD::workerLoop()
{
    std::vector<float> chunk(kChunkSize);

    while (_running.load(std::memory_order_acquire))
    {
        if (_ring.availableRead() >= kChunkSize)
        {
            _ring.read(chunk.data(), kChunkSize);

            float probability = runInference(chunk.data(), kChunkSize);
            _probability.store(probability, std::memory_order_release);

            if (probability >= kThreshold)
            {
                _isSpeech.store(true, std::memory_order_release);
                _hangoverRemaining.store(kHangoverMs, std::memory_order_relaxed);
            }
            else
            {
                int remaining = _hangoverRemaining.load(std::memory_order_relaxed);
                if (remaining > 0)
                {
                    remaining -= 32; // 32 ms per chunk
                    _hangoverRemaining.store(remaining, std::memory_order_relaxed);
                    if (remaining <= 0)
                        _isSpeech.store(false, std::memory_order_release);
                }
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

float SileroVAD::runInference(const float* data, size_t numSamples)
{
    if (!_modelLoaded.load(std::memory_order_acquire) || _session == nullptr)
        return 0.0f;

    std::array<int64_t, 2> audioShape = {1, static_cast<int64_t>(numSamples)};
    std::array<int64_t, 3> stateShape = {2, 1, 64};
    std::array<int64_t, 1> srShape = {1};

    auto audioTensor = Ort::Value::CreateTensor<float>(
        _memInfo, const_cast<float*>(data), numSamples,
        audioShape.data(), audioShape.size());
    auto hTensor = Ort::Value::CreateTensor<float>(
        _memInfo, _hState.data(), _hState.size(),
        stateShape.data(), stateShape.size());
    auto cTensor = Ort::Value::CreateTensor<float>(
        _memInfo, _cState.data(), _cState.size(),
        stateShape.data(), stateShape.size());
    auto srTensor = Ort::Value::CreateTensor<int64_t>(
        _memInfo, &_srState, 1, srShape.data(), srShape.size());

    const char* inputNames[] = {"input", "h", "c", "sr"};
    const char* outputNames[] = {"output", "hn", "cn"};

    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(std::move(audioTensor));
    inputs.push_back(std::move(hTensor));
    inputs.push_back(std::move(cTensor));
    inputs.push_back(std::move(srTensor));

    auto results = _session->Run(Ort::RunOptions{nullptr},
                                  inputNames, inputs.data(), inputs.size(),
                                  outputNames, 3);

    const float prob = results[0].GetTensorData<float>()[0];

    const float* hnData = results[1].GetTensorData<float>();
    const float* cnData = results[2].GetTensorData<float>();
    std::memcpy(_hState.data(), hnData, _hState.size() * sizeof(float));
    std::memcpy(_cState.data(), cnData, _cState.size() * sizeof(float));

    return prob;
}
