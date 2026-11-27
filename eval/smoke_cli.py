#!/usr/bin/env python3
"""End-to-end smoke test for CabinVoiceBenchCLI (macOS: uses `say` + `afconvert`).

Builds a noisy test clip, runs every mode, and checks that the output has the
input's length, is aligned at lag 0, and that Wiener at strength 0 and Bypass
reproduce the input. Prints the VAD segments found on the input.

    .venv/bin/python eval/smoke_cli.py build/CabinVoiceBenchCLI_artefacts/Release/CabinVoiceBenchCLI
"""
import json, subprocess, sys, tempfile, pathlib
import numpy as np, soundfile as sf
from scipy.signal import lfilter

cli = sys.argv[1]
model = str(pathlib.Path(__file__).resolve().parents[1] / "models" / "silero_vad.onnx")
tmp = pathlib.Path(tempfile.mkdtemp(prefix="cvb_smoke_"))

subprocess.run(["say", "-o", str(tmp / "s.aiff"), "Hey, navigate to the nearest charging station. "
                "Then turn the temperature down two degrees."], check=True)
subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@48000", "-c", "1",
                str(tmp / "s.aiff"), str(tmp / "s.wav")], check=True)
sp, fs = sf.read(tmp / "s.wav")
assert fs == 48000

rng = np.random.default_rng(0)
lead = int(1.5 * fs)
speech = np.concatenate([np.zeros(lead), sp, np.zeros(fs)])
noise = lfilter([1], [1, -0.98], rng.standard_normal(len(speech)))   # brown-ish, car-like
noise *= np.sqrt(np.mean(sp ** 2) / (np.mean(noise ** 2) * 10 ** (5 / 10)))  # 5 dB SNR
x = (speech + noise) * 0.5
sf.write(tmp / "noisy.wav", x, fs, subtype="PCM_24")

def run(mode, s):
    out, ev = tmp / f"{mode}_{s}.wav", tmp / f"{mode}_{s}.json"
    r = subprocess.run([cli, "--mode", mode, "--strength", str(s), "--input", str(tmp / "noisy.wav"),
                        "--output", str(out), "--vad-model", model, "--vad-events", str(ev)],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    return sf.read(out)[0], json.load(open(ev))

def lag(a, b):
    seg = slice(len(a) // 4, 3 * len(a) // 4)
    return max(range(-50, 51), key=lambda L: np.dot(a[seg], np.roll(b, -L)[seg]))

ok = True
for mode, s in [("bypass", 1.0), ("wiener", 0.0), ("wiener", 1.0), ("wiener", 0.5), ("rnnoise", 1.0)]:
    y, ev = run(mode, s)
    L = lag(x, y)
    exact = mode == "bypass" or (mode == "wiener" and s == 0.0)
    err = float(np.max(np.abs(y - x))) if exact else None
    print(f"{mode:8s} s={s:<4} len_ok={len(y) == len(x)} lag={L:+d} latency={ev['latency_samples']}"
          + (f" max|y-x|={err:.1e}" if err is not None else "")
          + f" in_segments={[[round(a, 2), round(b, 2)] for a, b in ev['input']['segments']]}"
          + f" out_onsets={ev['output']['num_onsets']}")
    ok &= len(y) == len(x) and L == 0 and (err is None or err < 1e-4)
print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
