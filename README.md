# CabinVoice Bench

Real-time, single-microphone speech enhancement front-end for in-car voice assistants, and a measurement harness that follows each front-end choice from the microphone to the recognizer and the turn-taking decision.

```
Mic ─► [ CabinVoice Bench: Bypass | Wiener (strength s) | RNNoise ] ─► ASR (Whisper) ─► transcript
                     │                                   │
               DNSMOS P.835 · PESQ · STOI           WER ± 95% CI
               measured latency · block time        false VAD triggers/min (turn-taking)
```

## The question

> **Which front-end makes a voice assistant hear correctly most often in car noise, and what does it cost in latency and CPU?**

Noise suppression is usually tuned for human listeners. Voice assistants fail on machine decisions: road noise trips the VAD (a false barge-in or wake), and a suppressor that sounds cleaner can remove detail the recognizer needs. Chondhekar et al. (2025) report enhancement raising semantic WER by 1.1 to 46.6 points absolute with MetricGAN+, and vendor tests (Krisp) report suppressed audio roughly doubling Whisper's WER while helping the VAD.

The working hypothesis is that **a voice front-end needs two different "clean" signals**: aggressive suppression for turn-taking, little or none for recognition. CabinVoice Bench measures that trade-off on recorded in-car noise and tests a split-path front-end. It is a reproducible replication and extension of a known effect, not a new discovery.

## Modes

| Mode | Algorithm | Latency (measured) | What it tests |
|------|-----------|--------------------|---------------|
| **Bypass** | None | 0 samples | Baseline |
| **Wiener** | Decision-directed a priori SNR (Ephraim & Malah 1984), α = 0.98, gain floor −26 dB, noise floor updated only when the VAD says silence | 1023 samples = 21.31 ms | Classical DSP |
| **Wiener, strength s** | Same filter; applied gain G^s, s ∈ [0, 1]. The SNR estimator always tracks the full-strength estimate, so only the output changes | 21.31 ms | Suppression-strength curve (WER vs s) |
| **RNNoise** | Hybrid DSP + GRU, 22 bands (Valin 2018), 480-sample frames, 48 kHz only | 959 samples = 19.98 ms | Neural library integration |

Latencies are measured by `Tests/TestLatency.cpp` (cross-correlation, irregular host block sizes) and must equal what each mode reports to the host. RNNoise's 959 samples are the FIFO (479) plus RNNoise's own analysis/synthesis overlap (480).

## Architecture

```
Input (48 kHz; stereo is folded to mono, output copied to all channels)
  │
  ├──► Downsampler (8th-order Butterworth LPF @ 6.4 kHz, fractional decimation)
  │      └─► SPSC ring ──► Worker thread: Silero VAD v5 (ONNX Runtime, 16 kHz, 32 ms chunks)
  │                           └─► atomic probability + speech flag (300 ms hangover)
  │                                 (gates Wiener noise-floor updates)
  │
  ├──► Mode switch
  │      ├─ Bypass:  pass-through
  │      ├─ Wiener:  STFT (1024-pt, sqrt-Hann, 50% overlap) ─► DD-SNR gains ^ s ─► iSTFT/OLA
  │      └─ RNNoise: FIFO (480) ─► rnnoise_process_frame   (pass-through if sr ≠ 48 kHz)
  │
  └──► Output gain ─► metering (RMS, peak via atomics)
```

The VAD is a side-chain: audio never passes through it, and inference never runs on the audio thread.

**Real-time constraints:**
- No heap allocations, locks or I/O in `processBlock` (templated callbacks, pre-allocated buffers)
- Lock-free SPSC ring and atomics between the audio and VAD threads
- ONNX inference on a worker thread; exceptions are caught there and disable the VAD instead of terminating the host
- `ScopedNoDenormals`; latency changes reported from the message thread (`AsyncUpdater`)

## Build

```bash
brew install onnxruntime          # macOS; elsewhere pass -DONNXRUNTIME_ROOT=<prefix>
cd cabinvoice-bench
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

**Requirements:** CMake 3.22+, a C++17 compiler, ONNX Runtime (tested with 1.32.0 from Homebrew). JUCE 8.0.4 and RNNoise v0.1.1 are fetched automatically.

ONNX Runtime is found locally rather than downloaded: the `microsoft/onnxruntime` GitHub release URLs stopped resolving in Nov 2026.

**Targets:**
- `CabinVoiceBench`: Standalone, VST3, AU
- `CabinVoiceBenchCLI`: offline WAV processor used by the eval harness
- `TestSTFT`, `TestLatency`, `TestSileroVAD`: run by `ctest`

### VAD model

The Silero VAD v5 model ships in `models/silero_vad.onnx` (MIT). The plugin looks for it, in order: next to the binary, in the bundle's `Resources/`, in `~/Library/Application Support/CabinVoice Bench/`, then in the source tree (dev builds). If it is missing the plugin still runs, shows "VAD off" in the editor, and the Wiener noise floor never updates.

## Offline CLI

```bash
./build/CabinVoiceBenchCLI_artefacts/Release/CabinVoiceBenchCLI \
    --mode wiener --strength 0.5 \
    --input noisy_48k.wav --output out/wiener_s0.5/noisy.wav \
    --vad-events out/wiener_s0.5/noisy.vad.json
```

- **Same DSP as the plugin:** same objects, parameters, block size (default 512) and Silero gating. The VAD flag for a block is the decision of the last 32 ms chunk ending at or before it, i.e. the plugin with zero inference delay.
- **Latency-aligned output:** output sample n corresponds to input sample n, with the same length, in every mode, so PESQ/STOI/segSNR are not biased against the filters.
- **48 kHz input only.** Resample upstream with a zero-phase resampler; the eval harness does this.
- `--vad-events` writes per-chunk probabilities and speech segments for both the input and the processed output. The output track is what a VAD placed after the filter would see (split-path analysis).
- `--no-vad` falls back to treating the first 500 ms as noise (legacy behaviour).
- `--mode whisper-tuned` is kept only for older result files (α = 0.92, floor −20 dB).

## Evaluation (planned, see status)

| Layer | Metric |
|-------|--------|
| Audio | DNSMOS P.835 (SIG / BAK / OVRL), PESQ and STOI as legacy proxies, latency-aligned |
| Cost | Measured algorithmic latency, mean and worst-case block time |
| Recognition | Normalised corpus WER ± bootstrap 95% CI (Whisper, temperature 0) |
| Turn-taking | False VAD onsets per minute on noise-only audio; onset delay on speech |

**Conditions:** clean speech (VoiceBank+DEMAND clean set, a SLURP subset) mixed with recorded in-car noise (DEMAND `TCAR`, ETSI EG 202 396-1 vehicle noises) at controlled SNRs, the EG 202 396 approach of recorded vehicle noise plus lab speech. A real-acoustics check replays the car noise through loudspeakers while speech is recorded on a single microphone.

**Front-ends compared:** Bypass, Wiener at s ∈ {0.25, 0.5, 1}, RNNoise, and a split path (full-strength filter → VAD, Bypass or s = 0.25 → ASR).

## Status

| Component | State |
|-----------|-------|
| STFT/OLA engine | Done; reconstruction and latency tests pass |
| Wiener + strength, RNNoise (48 kHz guard, measured latency) | Done; latency test passes |
| Silero VAD v5 (worker thread, anti-aliased downsampler) | Done; model test passes |
| JUCE plugin (Standalone, VST3, AU) | Being rebuilt after the VAD/parameter changes; pluginval pending |
| `CabinVoiceBenchCLI` (latency-aligned, VAD-gated, VAD events) | Written; build pending |
| Eval harness (normalised WER + CIs, DNSMOS, noise mixing) | Next |
| In-car noise experiment, split path | Planned |

## Project structure

```
cabinvoice-bench/
├── CMakeLists.txt
├── Source/
│   ├── Plugin/        PluginProcessor (mode switch, strength, mono fold, metering), PluginEditor (mode, strength, VAD meter, status)
│   ├── Parameters/    ParameterIDs (mode v2, strength, gains), ParameterLayout
│   ├── DSP/
│   │   ├── STFT/STFTEngine.h          1024-pt STFT/OLA, sqrt-Hann
│   │   ├── Wiener/WienerSuppressor.h  DD-SNR, gain floor, strength
│   │   ├── RNNoise/RNNoiseWrapper.*   RNNoise v0.1.1 + FIFO, 48 kHz only
│   │   ├── VAD/SileroModel.*          synchronous Silero v5 inference + hangover (CLI, tests, worker)
│   │   ├── VAD/SileroVAD.*            real-time wrapper: ring + worker thread
│   │   ├── VAD/Downsampler.h          host rate → 16 kHz, anti-aliased, RT-safe
│   │   └── Common/                    SPSCRingBuffer, FIFOAdapter
│   └── CLI/BlockProcessor.cpp         offline processor
├── Tests/             TestSTFTPerfectReconstruction, TestLatency, TestSileroVAD
├── eval/              download script, evaluate.py, requirements
└── models/            silero_vad.onnx
```

## Limitations

Single microphone (no beamforming), no acoustic echo cancellation, no calibrated SPL, and in-car noise comes from public recordings rather than a vehicle under test.

## Key references

- Ephraim, Y. & Malah, D. (1984). "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator." IEEE Trans. ASSP.
- Valin, J.-M. (2018). "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement." IEEE MMSP.
- Chondhekar, S. et al. (2025). "When De-noising Hurts: A Systematic Study of Speech Enhancement Effects on Modern Medical ASR Systems." arXiv:2512.17562.
- Reddy, C. K. A., Gopal, V. & Cutler, R. (2022). "DNSMOS P.835." ICASSP.
- Valentini-Botinhao, C. et al. (2016). "Speech Enhancement for a Noise-Robust Text-to-Speech Synthesis System." Interspeech.
- Thiemann, J., Ito, N. & Vincent, E. (2013). "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)."
- Bastianelli, E. et al. (2020). "SLURP: A Spoken Language Understanding Resource Package." EMNLP.
- ETSI EG 202 396-1: background noise simulation for speech quality testing.
- Silero VAD: https://github.com/snakers4/silero-vad

## License

Portfolio project by Italo Rojas. Not licensed for commercial use. SLURP audio is CC BY-NC 4.0 and is not redistributed here.
