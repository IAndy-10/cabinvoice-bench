#pragma once

#include "../STFT/STFTEngine.h"
#include <complex>
#include <vector>

/// Per-bin Wiener suppressor with decision-directed a priori SNR estimation.
/// Ephraim & Malah (1984). Noise floor updated during VAD-silence frames.
///
/// G(k) = SNR_priori(k) / (1 + SNR_priori(k))
/// SNR_priori(k) = alpha * (|X_prev(k)|^2 / noise_var(k)) + (1 - alpha) * max(SNR_post(k) - 1, 0)
class WienerSuppressor {
public:
    struct Config {
        float alpha     = 0.98f;  // DD smoothing
        float gainFloor = 0.05f;  // -26 dB minimum gain
    };

    void prepare(size_t numBins, Config cfg) {
        _numBins = numBins;
        _config = cfg;
        _noisePower.assign(numBins, 1e-6f);
        _prevMagnitude.assign(numBins, 0.0f);
        _snrPriori.assign(numBins, 0.0f);
        _noiseEstimateFrames = 0;
    }

    void setConfig(Config cfg) { _config = cfg; }

    /// Process one STFT frame in-place. Called from STFTEngine's spectral callback.
    void process(std::complex<float>* bins, size_t numBins, bool isSpeech) {
        for (size_t k = 0; k < numBins; ++k) {
            const float mag = std::abs(bins[k]);
            const float magSq = mag * mag;

            // Update noise estimate during silence
            if (!isSpeech) {
                const float beta = (_noiseEstimateFrames < 20) ? 0.5f : 0.95f;
                _noisePower[k] = beta * _noisePower[k] + (1.0f - beta) * magSq;
            }

            // Posterior SNR
            const float snrPost = magSq / std::max(_noisePower[k], 1e-10f);

            // Decision-directed a priori SNR
            const float prevSNR = (_prevMagnitude[k] * _prevMagnitude[k])
                                  / std::max(_noisePower[k], 1e-10f);
            _snrPriori[k] = _config.alpha * prevSNR
                            + (1.0f - _config.alpha) * std::max(snrPost - 1.0f, 0.0f);

            // Wiener gain
            float gain = _snrPriori[k] / (1.0f + _snrPriori[k]);
            gain = std::max(gain, _config.gainFloor);

            bins[k] *= gain;
            _prevMagnitude[k] = std::abs(bins[k]);
        }

        if (!isSpeech)
            ++_noiseEstimateFrames;
    }

    void reset() {
        std::fill(_noisePower.begin(), _noisePower.end(), 1e-6f);
        std::fill(_prevMagnitude.begin(), _prevMagnitude.end(), 0.0f);
        std::fill(_snrPriori.begin(), _snrPriori.end(), 0.0f);
        _noiseEstimateFrames = 0;
    }

private:
    Config _config;
    size_t _numBins = 0;
    std::vector<float> _noisePower;
    std::vector<float> _prevMagnitude;
    std::vector<float> _snrPriori;
    int _noiseEstimateFrames = 0;
};
