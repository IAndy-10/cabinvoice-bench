#pragma once

#include <cmath>
#include <cstddef>

/// Streaming host-rate -> 16 kHz converter for the VAD side-chain.
/// 8th-order Butterworth low-pass at 6.4 kHz (four RBJ biquads), then
/// linear-interpolated fractional decimation. Filter state and output phase
/// carry across blocks, so block boundaries leave no trace.
/// Real-time safe: no allocation, no locks. Not a reference-quality resampler;
/// it only needs to keep aliasing out of the VAD's input band.
class Downsampler {
public:
    void prepare(double inputRate, double outputRate = 16000.0) {
        _step = inputRate / outputRate;
        // 0.40 * fs_out keeps the VAD band (speech energy < ~6 kHz) and puts
        // >40 dB of rejection on anything that would fold back into it.
        const double fc = 0.40 * outputRate;
        // Butterworth 8th order = four biquads with these Qs.
        const double q[kNumStages] = { 0.50979558, 0.60134489, 0.89997622, 2.56291545 };
        for (int i = 0; i < kNumStages; ++i)
            _s[i].design(fc, inputRate, q[i]);
        reset();
    }

    void reset() {
        for (auto& st : _s) st.clear();
        _prev = 0.0;
        _t = 1.0;
    }

    /// Calls emit(float) once per 16 kHz output sample.
    template <typename Emit>
    void process(const float* in, size_t numSamples, Emit&& emit) {
        for (size_t n = 0; n < numSamples; ++n) {
            double y = static_cast<double>(in[n]);
            for (auto& st : _s) y = st.tick(y);
            while (_t <= 1.0) {
                emit(static_cast<float>(_prev + (y - _prev) * _t));
                _t += _step;
            }
            _t -= 1.0;
            _prev = y;
        }
    }

private:
    struct Biquad {
        double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
        double z1 = 0, z2 = 0;

        void design(double fc, double fs, double q) {
            const double w0 = 2.0 * 3.14159265358979323846 * fc / fs;
            const double cw = std::cos(w0), alpha = std::sin(w0) / (2.0 * q);
            const double a0 = 1.0 + alpha;
            b0 = (1.0 - cw) / 2.0 / a0;
            b1 = (1.0 - cw) / a0;
            b2 = b0;
            a1 = -2.0 * cw / a0;
            a2 = (1.0 - alpha) / a0;
        }
        void clear() { z1 = z2 = 0.0; }
        double tick(double x) { // transposed direct form II
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    static constexpr int kNumStages = 4;
    Biquad _s[kNumStages];
    double _step = 3.0;
    double _prev = 0.0;
    double _t = 1.0;   // position of next output between _prev (0) and current input (1)
};
