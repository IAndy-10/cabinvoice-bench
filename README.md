# Clean Ears

Real-time, single-microphone speech enhancement front-end for voice agents — and a measurement harness that follows each front-end choice all the way from the microphone to the agent's decision.

```
Mic ─► [ Clean Ears: Bypass | Wiener | RNNoise | ASR-tuned Wiener ] ─► ASR ─► Turn + intent decisions
                          │                                            │                │
                    PESQ · STOI · segSNR                     WER (Scribe, Whisper)   endpointing · false barge-ins · intent F1 · ECE
                    latency · CPU
```

## The Question

> **Which front-end makes a voice agent decide correctly most often, and what does it cost in latency and CPU?**

Noise suppression is usually tuned for human ears (PESQ, STOI). Voice agents fail on machine decisions: a TV triggers a false barge-in, noise breaks endpointing, a muffled word flips the intent. Recent work ("When De-noising Hurts", Chondhekar et al. 2025) shows enhancement can raise ASR word error rate by up to 46% because modern recognizers are already noise-robust and suppression removes detail they rely on. Clean Ears measures whether that carries through to agent-level decisions — and whether the answer differs between a cloud ASR (ElevenLabs Scribe) and a local one (Whisper).

The interesting result is any row where audio quality goes up but decision accuracy goes down.

## Suppressor Modes

| Mode | Algorithm | Algorithmic latency | What it tests |
|------|-----------|---------------------|---------------|
| **Bypass** | None | 0 ms | Baseline |
| **Wiener** | Decision-directed SNR (Ephraim & Malah 1984), α = 0.98, gain floor −26 dB | 21.3 ms | Classical DSP |
| **RNNoise** | Hybrid DSP + GRU, 22 bands (Valin 2018), 480-sample frames | 10 ms | Neural library integration |
| **Wiener (ASR-tuned)** | Same filter, α = 0.92, gain floor −20 dB | 21.3 ms | Front-end tuned for machines, not ears |

The ASR-tuned variant (CLI flag `whisper-tuned`) is deliberately less aggressive: worse PESQ/STOI, potentially lower WER. Finding where those diverge is the point.

**Optional cloud reference:** ElevenLabs Voice Isolator, run offline on the same clips, as an upper bound on suppression quality (not real-time, not local).

## Architecture

```
Input (mono, 48 kHz)
  │
  ├──► SPSC ring ──► Worker thread: Silero VAD (ONNX Runtime, 16 kHz)
  │                      └─► atomic speech flag + hangover
  │                            (gates noise-floor updates; emits speech start/stop events)
  │
  ├──► Mode switch
  │      ├─ Bypass:     pass-through
  │      ├─ Wiener:     STFT (1024-pt, sqrt-Hann, 50% overlap) ─► DD-SNR gains ─► iSTFT/OLA
  │      ├─ RNNoise:    FIFO (480 samples) ─► rnnoise_process_frame
  │      └─ ASR-tuned:  same STFT path, gentler Wiener parameters
  │
  └──► Output ─► metering (RMS, peak via atomics)
```

The VAD is a side-chain: audio never passes through it.

**Real-time constraints enforced:**
- No heap allocations in `processBlock` (templated callbacks, pre-allocated buffers)
- No locks (SPSC ring buffer, atomics for the VAD flag)
- No I/O or inference on the audio thread (ONNX runs on the worker thread)
- Denormal flushing (`ScopedNoDenormals`)
- Latency changes reported from the message thread (`AsyncUpdater`)

ASR, turn detection and intent classification are network or model calls on text/audio *after* Clean Ears — never in the audio path.

## Build

```bash
cd clean-ears
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

**Requirements:** CMake 3.22+, C++17 compiler, macOS (primary) or Linux/Windows.

Dependencies are fetched automatically: JUCE 8.0.4, ONNX Runtime 1.17.1, RNNoise v0.1.1.

**Targets:**
- `CleanEars` — Standalone, VST3, AU (`build/CleanEars_artefacts/Release/…`)
- `CleanEarsCLI` — offline WAV block processor
- `TestSTFT` — perfect-reconstruction test (all gains = 1 must reproduce the input)

### VAD model

The Silero VAD model ships in `models/`. Copy it next to the executable:

```bash
cp models/silero_vad.onnx build/CleanEars_artefacts/Release/Standalone/
```

Without it the plugin still runs, but the Wiener noise estimate updates continuously instead of only during silence.

## Evaluation

### Datasets

| Dataset | Used for | Notes |
|---------|----------|-------|
| **VoiceBank+DEMAND** (Valentini-Botinhao et al. 2016) | Audio quality (PESQ, STOI, segSNR), WER | 824-utterance test set, 48 kHz, noisy/clean pairs |
| **SLURP** (Bastianelli et al. 2020) | Decisions: intent, endpointing, false barge-ins | ~100 close-talk (`-headset`) utterances with intent labels, mixed with DEMAND noise; far-field recordings of the same sentences as a real-room condition. Audio is CC BY-NC 4.0 — downloaded locally, never committed. |

Test conditions for the SLURP subset: stationary noise (10 dB SNR), non-stationary noise (5 and 10 dB SNR), and reverberation (simulated RIR, T60 ≈ 400 ms) — reported separately, since no mode dereverberates.

### Decision layer

| Decision | How it's measured |
|----------|-------------------|
| **Endpointing** | Pipecat Smart Turn v3 `P(complete)` on each clip; truncated clips supply the "incomplete" class. Accuracy, ECE, and median delay from true speech end to turn commit. |
| **False barge-ins** | Noise-only and background-talker segments inserted between utterances; count speech/turn onsets per minute. |
| **Intent** | LLM classifier over the SLURP intent set from each ASR transcript; micro-F1 on scenario and action. |

### Metrics

| Layer | Metric | Tool |
|-------|--------|------|
| Audio quality | PESQ, STOI, segmental SNR | `pesq`, `pystoi` |
| Real-time cost | CPU %, worst-case block time, algorithmic latency | In-callback timers |
| Recognition | WER (normalized text) | ElevenLabs Scribe v2 (batch), Whisper (local), `jiwer` |
| Decisions | Endpointing accuracy + delay, false barge-ins/min, intent F1, ECE | Smart Turn v3, LLM intent classifier |

ElevenLabs results are cached to disk per clip and mode, so each clip is only sent once.

### Running it

```bash
# Download VoiceBank+DEMAND (+ reference transcripts) into eval/data/
./eval/download_voicebank_demand.sh

# Process the noisy set with each mode
./build/CleanEarsCLI_artefacts/Release/clean-ears-cli \
    --mode wiener --input noisy.wav --output out/wiener/noisy.wav

# Score the enhanced directories
pip install -r eval/requirements.txt
python eval/evaluate.py \
    --noisy-dir  eval/data/voicebank_demand/noisy_testset_wav \
    --clean-dir  eval/data/voicebank_demand/clean_testset_wav \
    --enhanced-dirs wiener=out/wiener whisper-tuned=out/whisper_tuned rnnoise=out/rnnoise \
    --ref-transcripts eval/data/voicebank_demand/testset_txt \
    --whisper-model base \
    --output-json results/eval_results.json
```

### Headline deliverable

One table, one row per mode, from audio quality through decisions — repeated for each ASR:

| Mode | PESQ | STOI | WER (Scribe) | WER (Whisper) | Endpoint acc | False barge-ins/min | Intent F1 | Latency | CPU % |
|------|------|------|--------------|---------------|--------------|---------------------|-----------|---------|-------|
| Bypass | — | — | — | — | — | — | — | 0 ms | — |
| Wiener | — | — | — | — | — | — | — | 21.3 ms | — |
| RNNoise | — | — | — | — | — | — | — | 10 ms | — |
| ASR-tuned | — | — | — | — | — | — | — | 21.3 ms | — |

### Live demo

A Pipecat voice agent with Clean Ears as its audio input filter, ElevenLabs for speech-to-text and text-to-speech, and the suppressor mode switchable mid-conversation.

## Status

| Component | State |
|-----------|-------|
| STFT/OLA engine + reconstruction test | Done |
| Wiener, ASR-tuned Wiener, RNNoise | Done |
| JUCE plugin (Standalone, VST3, AU) | Builds; pluginval pending |
| Silero VAD worker thread | Done |
| `CleanEarsCLI` | Build error — being fixed |
| Eval harness: PESQ, STOI, Whisper WER | Done |
| Eval harness: CLI driver, segSNR | Being merged in |
| ElevenLabs Scribe, SLURP corpus, decision layer | Planned |
| Pipecat live demo | Planned |

## Project Structure

```
clean-ears/
├── CMakeLists.txt
├── README.md
├── Source/
│   ├── Plugin/
│   │   ├── PluginProcessor.h/cpp     # 4-mode switch, gains, metering
│   │   └── PluginEditor.h/cpp        # Mode selector UI
│   ├── Parameters/
│   │   ├── ParameterIDs.h            # mode, inputGain, outputGain
│   │   └── ParameterLayout.h         # APVTS layout
│   ├── DSP/
│   │   ├── STFT/STFTEngine.h         # 1024-pt STFT/OLA, sqrt-Hann
│   │   ├── Wiener/WienerSuppressor.h # DD-SNR, configurable alpha/gainFloor
│   │   ├── RNNoise/RNNoiseWrapper.*  # RNNoise v0.1.1 + FIFO
│   │   ├── VAD/SileroVAD.*           # ONNX Runtime, worker thread
│   │   └── Common/
│   │       ├── SPSCRingBuffer.h      # Lock-free ring buffer
│   │       └── FIFOAdapter.h         # Variable-to-fixed block adapter
│   └── CLI/
│       └── BlockProcessor.cpp        # Offline WAV processor
├── Tests/
│   └── TestSTFTPerfectReconstruction.cpp
├── eval/
│   ├── download_voicebank_demand.sh
│   ├── evaluate.py                   # PESQ / STOI / WER harness
│   └── requirements.txt
└── models/
    └── silero_vad.onnx
```

## Key References

- Ephraim, Y. & Malah, D. (1984). "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator." IEEE Trans. ASSP.
- Valin, J.-M. (2018). "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement." IEEE MMSP.
- Chondhekar, S. et al. (2025). "When De-noising Hurts: A Systematic Study of Speech Enhancement Effects on Modern Medical ASR Systems." arXiv:2512.17562.
- Valentini-Botinhao, C. et al. (2016). "Speech Enhancement for a Noise-Robust Text-to-Speech Synthesis System." Interspeech.
- Bastianelli, E. et al. (2020). "SLURP: A Spoken Language Understanding Resource Package." EMNLP.
- Silero VAD: https://github.com/snakers4/silero-vad
- Pipecat Smart Turn: https://github.com/pipecat-ai/smart-turn

## License

Portfolio project by Italo Rojas. Not licensed for commercial use. SLURP audio is CC BY-NC 4.0 and is not redistributed here.
