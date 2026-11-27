#pragma once

#include <juce_dsp/juce_dsp.h>
#include <complex>
#include <vector>

/// Real-time STFT/OLA engine with sqrt-Hann windows for perfect reconstruction.
/// Frame size: 1024 samples (21.3 ms at 48 kHz), 50% overlap (hop = 512).
/// Note: designed for 48 kHz operation. At other sample rates the frame duration changes.
class STFTEngine {
public:
    static constexpr size_t kFFTOrder    = 10;           // 2^10 = 1024
    static constexpr size_t kFrameSize   = 1 << kFFTOrder; // 1024
    static constexpr size_t kHopSize     = kFrameSize / 2; // 512
    static constexpr size_t kNumBins     = kFrameSize / 2 + 1; // 513

    STFTEngine() : _fft(kFFTOrder) {}

    void prepare(double /*sampleRate*/) {
        _inputBuffer.resize(kFrameSize, 0.0f);
        _fftData.resize(kFrameSize * 2, 0.0f);
        _overlapBuffer.resize(kFrameSize, 0.0f);
        _inputWritePos = 0;
        _outputReadPos = 0;
        _samplesUntilNextFrame = kHopSize;

        buildSqrtHannWindow();
    }

    /// Process a block of mono audio through STFT -> callback -> iSTFT/OLA.
    /// Callback is templated to avoid std::function heap allocation on audio thread.
    template <typename Callback>
    void process(float* data, size_t numSamples, Callback&& callback) {
        for (size_t i = 0; i < numSamples; ++i) {
            _inputBuffer[_inputWritePos] = data[i];
            _inputWritePos = (_inputWritePos + 1) % kFrameSize;
            --_samplesUntilNextFrame;

            if (_samplesUntilNextFrame == 0) {
                processFrame(callback);
                _samplesUntilNextFrame = kHopSize;
            }

            data[i] = _overlapBuffer[_outputReadPos];
            _overlapBuffer[_outputReadPos] = 0.0f;
            _outputReadPos = (_outputReadPos + 1) % kFrameSize;
        }
    }

    size_t getLatencySamples() const { return kFrameSize; }

    void reset() {
        std::fill(_inputBuffer.begin(), _inputBuffer.end(), 0.0f);
        std::fill(_fftData.begin(), _fftData.end(), 0.0f);
        std::fill(_overlapBuffer.begin(), _overlapBuffer.end(), 0.0f);
        _inputWritePos = 0;
        _outputReadPos = 0;
        _samplesUntilNextFrame = kHopSize;
    }

private:
    void buildSqrtHannWindow() {
        _window.resize(kFrameSize);
        for (size_t i = 0; i < kFrameSize; ++i) {
            float hann = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi
                                                  * static_cast<float>(i)
                                                  / static_cast<float>(kFrameSize)));
            _window[i] = std::sqrt(hann);
        }
    }

    template <typename Callback>
    void processFrame(Callback&& callback) {
        // Copy input with analysis window
        for (size_t i = 0; i < kFrameSize; ++i) {
            auto readIdx = (_inputWritePos + i) % kFrameSize;
            _fftData[i] = _inputBuffer[readIdx] * _window[i];
        }
        std::fill(_fftData.begin() + kFrameSize, _fftData.end(), 0.0f);

        _fft.performRealOnlyForwardTransform(_fftData.data(), true);

        callback(reinterpret_cast<std::complex<float>*>(_fftData.data()), kNumBins);

        _fft.performRealOnlyInverseTransform(_fftData.data());

        // Apply synthesis window and overlap-add
        for (size_t i = 0; i < kFrameSize; ++i) {
            auto writeIdx = (_outputReadPos + i) % kFrameSize;
            _overlapBuffer[writeIdx] += _fftData[i] * _window[i];
        }
    }

    juce::dsp::FFT _fft;
    std::vector<float> _window;
    std::vector<float> _inputBuffer;
    std::vector<float> _fftData;
    std::vector<float> _overlapBuffer;
    size_t _inputWritePos = 0;
    size_t _outputReadPos = 0;
    size_t _samplesUntilNextFrame = kHopSize;
};
