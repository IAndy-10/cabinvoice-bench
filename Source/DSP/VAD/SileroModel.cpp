#include "SileroModel.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace {

std::vector<std::string> names(Ort::Session& s, bool inputs)
{
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> out;
    const size_t n = inputs ? s.GetInputCount() : s.GetOutputCount();
    for (size_t i = 0; i < n; ++i)
        out.emplace_back(inputs ? s.GetInputNameAllocated(i, alloc).get()
                                : s.GetOutputNameAllocated(i, alloc).get());
    return out;
}

bool contains(const std::vector<std::string>& v, const char* name)
{
    return std::find(v.begin(), v.end(), name) != v.end();
}

} // namespace

bool SileroModel::load(const std::string& modelPath, std::string& error)
{
    _session.reset();
    try
    {
        if (_env == nullptr)
            _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "silero_vad");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto session = std::make_unique<Ort::Session>(*_env, modelPath.c_str(), opts);

        const auto in  = names(*session, true);
        const auto out = names(*session, false);
        if (!contains(in, "input") || !contains(in, "state") || !contains(in, "sr")
            || !contains(out, "output") || !contains(out, "stateN"))
        {
            error = "model does not have the Silero v5 interface (input/state/sr -> output/stateN)";
            return false;
        }

        _session = std::move(session);
        reset();

        // Warm-up so the first real chunk does not pay graph-initialisation cost.
        std::array<float, kChunkSize> silence{};
        process(silence.data());
        reset();
        return true;
    }
    catch (const Ort::Exception& e)
    {
        _session.reset();
        error = e.what();
        return false;
    }
}

void SileroModel::reset() noexcept
{
    _input.fill(0.0f);
    _state.fill(0.0f);
}

float SileroModel::process(const float* chunk)
{
    // _input = [last 64 samples of previous chunk | this chunk]
    std::memcpy(_input.data() + kContextSize, chunk, kChunkSize * sizeof(float));

    const std::array<int64_t, 2> inputShape { 1, static_cast<int64_t>(_input.size()) };
    const std::array<int64_t, 3> stateShape { 2, 1, 128 };

    std::array<Ort::Value, 3> inputs {
        Ort::Value::CreateTensor<float>(_memInfo, _input.data(), _input.size(),
                                        inputShape.data(), inputShape.size()),
        Ort::Value::CreateTensor<float>(_memInfo, _state.data(), _state.size(),
                                        stateShape.data(), stateShape.size()),
        Ort::Value::CreateTensor<int64_t>(_memInfo, &_sr, 1, nullptr, 0) // scalar
    };

    static constexpr const char* inputNames[]  = { "input", "state", "sr" };
    static constexpr const char* outputNames[] = { "output", "stateN" };

    auto results = _session->Run(Ort::RunOptions{ nullptr },
                                 inputNames, inputs.data(), inputs.size(),
                                 outputNames, 2);

    const float prob = results[0].GetTensorData<float>()[0];
    std::memcpy(_state.data(), results[1].GetTensorData<float>(), kStateSize * sizeof(float));

    // Carry the tail of this chunk as next chunk's context.
    std::memmove(_input.data(), _input.data() + kChunkSize, kContextSize * sizeof(float));
    return prob;
}
