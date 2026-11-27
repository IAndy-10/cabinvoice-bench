#pragma once

#include <cstddef>
#include <vector>

/// Adapts variable-size host blocks to fixed-size processing frames.
/// Used for RNNoise (480 samples). No allocations after prepare().
/// Callback is templated to avoid std::function heap allocation on audio thread.
class FIFOAdapter {
public:
    FIFOAdapter() = default;

    void prepare(size_t frameSize, size_t maxBlockSize = 4096) {
        _frameSize = frameSize;
        _inputFIFO.resize(frameSize, 0.0f);
        // Output ring sized for max block + one extra frame
        _outputRingSize = frameSize + maxBlockSize;
        _outputRing.resize(_outputRingSize, 0.0f);
        _inputWritePos = 0;
        _outputWritePos = 0;
        _outputReadPos = 0;
        _firstFrameProcessed = false;
    }

    /// Push `numSamples` from `input`, process complete frames via `callback`,
    /// pull `numSamples` into `output`. Output is delayed by frameSize - 1 samples:
    /// the sample that completes a frame triggers processing and is output-read
    /// in the same iteration, so it emerges as the frame's first output sample.
    template <typename Callback>
    void process(const float* input, float* output, size_t numSamples, Callback&& callback) {
        for (size_t i = 0; i < numSamples; ++i) {
            _inputFIFO[_inputWritePos++] = input[i];

            if (_inputWritePos == _frameSize) {
                _inputWritePos = 0;
                // Write processed output into the output ring
                callback(_inputFIFO.data(), _outputRing.data() + _outputWritePos);
                _outputWritePos = (_outputWritePos + _frameSize) % _outputRingSize;
                _firstFrameProcessed = true;
            }

            if (_firstFrameProcessed) {
                output[i] = _outputRing[_outputReadPos];
                _outputRing[_outputReadPos] = 0.0f;
                _outputReadPos = (_outputReadPos + 1) % _outputRingSize;
            } else {
                output[i] = 0.0f;
            }
        }
    }

    size_t getLatencySamples() const { return _frameSize - 1; }

    void reset() {
        std::fill(_inputFIFO.begin(), _inputFIFO.end(), 0.0f);
        std::fill(_outputRing.begin(), _outputRing.end(), 0.0f);
        _inputWritePos = 0;
        _outputWritePos = 0;
        _outputReadPos = 0;
        _firstFrameProcessed = false;
    }

private:
    size_t _frameSize = 0;
    size_t _outputRingSize = 0;
    std::vector<float> _inputFIFO;
    std::vector<float> _outputRing;
    size_t _inputWritePos = 0;
    size_t _outputWritePos = 0;
    size_t _outputReadPos = 0;
    bool _firstFrameProcessed = false;
};
