#!/usr/bin/env python3
"""Evaluation pipeline for CabinVoice Bench speech enhancement.

Measures PESQ, STOI, and WER (via Whisper + jiwer) across enhancement modes
on the VoiceBank+DEMAND test set or any directory of noisy/clean WAV pairs.

Modes:
  - Bypass:         unprocessed noisy input (baseline)
  - Wiener:         decision-directed Wiener filter (alpha=0.98, gain_floor=-30 dB)
  - Whisper-tuned:  less aggressive Wiener (alpha=0.92, gain_floor=-12 dB)
  - RNNoise:        hybrid DSP/GRU (Valin 2018)

The "Whisper-tuned" variant tests the research hypothesis from
docs/research/clean-ears-viability.md: a perceptually *worse* enhancement
(higher gain floor, less suppression) may produce *lower* WER because it
preserves speech detail that Whisper needs.

Usage:
    python evaluate.py \\
        --noisy-dir  data/voicebank_demand/noisy_testset_wav \\
        --clean-dir  data/voicebank_demand/clean_testset_wav \\
        --enhanced-dirs wiener=out/wiener whisper-tuned=out/whisper_tuned rnnoise=out/rnnoise \\
        --whisper-model base \\
        --output-json results/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer
from pesq import pesq
from pystoi import stoi
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PESQ_SR = 16_000  # PESQ requires 16 kHz


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Mode(Enum):
    BYPASS = "Bypass"
    WIENER = "Wiener"
    WHISPER_TUNED = "Whisper-tuned"
    RNNOISE = "RNNoise"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FileResult:
    filename: str
    pesq_score: float
    stoi_score: float
    wer_score: float | None = None


@dataclass
class ModeResult:
    mode: str
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def mean_pesq(self) -> float:
        vals = [r.pesq_score for r in self.file_results if not np.isnan(r.pesq_score)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_stoi(self) -> float:
        vals = [r.stoi_score for r in self.file_results if not np.isnan(r.stoi_score)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_wer(self) -> float | None:
        vals = [r.wer_score for r in self.file_results if r.wer_score is not None]
        return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_wav(path: Path, target_sr: int = PESQ_SR) -> tuple[np.ndarray, int]:
    """Load a WAV file, convert to mono, and resample to target_sr."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import resampy

        data = resampy.resample(data, sr, target_sr)
    return data, target_sr


def align_lengths(
    ref: np.ndarray, deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trim both signals to the shorter length."""
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def compute_pesq_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    ref, deg = align_lengths(ref, deg)
    return float(pesq(sr, ref, deg, "wb"))


def compute_stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    ref, deg = align_lengths(ref, deg)
    return float(stoi(ref, deg, sr, extended=False))


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------
class WhisperTranscriber:
    """Lazy-loaded Whisper model for WER evaluation."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import whisper

        log.info("Loading Whisper model '%s' ...", self._model_name)
        self._model = whisper.load_model(self._model_name)
        log.info("Whisper model loaded.")

    def transcribe(self, audio_path: Path) -> str:
        self._load()
        result = self._model.transcribe(str(audio_path), language="en")
        return result["text"].strip()


def load_ref_transcripts(transcript_dir: Path) -> dict[str, str]:
    """Load per-file reference transcripts from <stem>.txt files."""
    transcripts: dict[str, str] = {}
    if not transcript_dir.is_dir():
        log.warning("Transcript directory not found: %s", transcript_dir)
        return transcripts
    for txt in sorted(transcript_dir.glob("*.txt")):
        transcripts[txt.stem] = txt.read_text().strip()
    return transcripts


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------
def evaluate_mode(
    mode: str,
    enhanced_dir: Path,
    clean_dir: Path,
    ref_transcripts: dict[str, str],
    transcriber: WhisperTranscriber | None,
    max_files: int | None,
) -> ModeResult:
    result = ModeResult(mode=mode)

    wav_files = sorted(enhanced_dir.glob("*.wav"))
    if not wav_files:
        log.warning("No WAV files in %s", enhanced_dir)
        return result

    if max_files is not None:
        wav_files = wav_files[:max_files]

    log.info(
        "Evaluating mode '%s': %d files from %s",
        mode,
        len(wav_files),
        enhanced_dir,
    )

    for i, wav_path in enumerate(wav_files, 1):
        stem = wav_path.stem
        clean_path = clean_dir / wav_path.name
        if not clean_path.exists():
            log.warning("No clean reference for %s — skipping.", stem)
            continue

        # Load audio at 16 kHz for PESQ/STOI
        try:
            ref, sr = load_wav(clean_path)
            deg, _ = load_wav(wav_path)
        except Exception:
            log.exception("Failed to load %s", stem)
            continue

        # PESQ
        try:
            p = compute_pesq_score(ref, deg, sr)
        except Exception:
            log.exception("PESQ failed for %s", stem)
            p = float("nan")

        # STOI
        try:
            s = compute_stoi_score(ref, deg, sr)
        except Exception:
            log.exception("STOI failed for %s", stem)
            s = float("nan")

        # WER via Whisper
        w: float | None = None
        ref_text = ref_transcripts.get(stem)
        if ref_text is not None and transcriber is not None:
            try:
                hyp_text = transcriber.transcribe(wav_path)
                if not ref_text:
                    w = 0.0 if not hyp_text else 1.0
                else:
                    w = float(compute_wer(ref_text, hyp_text))
            except Exception:
                log.exception("Whisper/WER failed for %s", stem)

        result.file_results.append(FileResult(stem, p, s, w))

        if i % 50 == 0:
            log.info("  [%s] %d / %d files done", mode, i, len(wav_files))

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def build_summary_table(results: list[ModeResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for r in results:
        row: dict[str, str] = {
            "Mode": r.mode,
            "PESQ": f"{r.mean_pesq:.3f}",
            "STOI": f"{r.mean_stoi:.4f}",
            "Files": str(len(r.file_results)),
        }
        wer_val = r.mean_wer
        row["WER"] = f"{wer_val:.3f}" if wer_val is not None else "—"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def parse_mode_path_pairs(raw: list[str]) -> dict[str, Path]:
    """Parse 'mode=path' pairs from CLI arguments."""
    pairs: dict[str, Path] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"Expected mode=path format, got '{item}'. "
                f"Example: wiener=out/wiener rnnoise=out/rnnoise"
            )
        mode, path_str = item.split("=", 1)
        pairs[mode] = Path(path_str)
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CabinVoice Bench speech enhancement across modes. "
            "Measures PESQ, STOI, and WER (Whisper + jiwer)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Audio-only metrics (no WER):\n"
            "  python evaluate.py \\\n"
            "    --noisy-dir data/vb/noisy_testset_wav \\\n"
            "    --clean-dir data/vb/clean_testset_wav \\\n"
            "    --enhanced-dirs wiener=out/wiener whisper-tuned=out/wt rnnoise=out/rn\n"
            "\n"
            "  # Full pipeline with Whisper WER:\n"
            "  python evaluate.py \\\n"
            "    --noisy-dir data/vb/noisy_testset_wav \\\n"
            "    --clean-dir data/vb/clean_testset_wav \\\n"
            "    --enhanced-dirs wiener=out/wiener whisper-tuned=out/wt rnnoise=out/rn \\\n"
            "    --ref-transcripts data/vb/transcripts \\\n"
            "    --whisper-model base \\\n"
            "    --output-json results/eval.json\n"
        ),
    )
    parser.add_argument(
        "--noisy-dir",
        type=Path,
        required=True,
        help="Directory of noisy input WAVs (used as Bypass baseline).",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        required=True,
        help="Directory of clean reference WAVs.",
    )
    parser.add_argument(
        "--enhanced-dirs",
        nargs="*",
        default=[],
        help=(
            "Enhanced output dirs as mode=path pairs. "
            "E.g.: wiener=out/wiener whisper-tuned=out/wt rnnoise=out/rn"
        ),
    )
    parser.add_argument(
        "--ref-transcripts",
        type=Path,
        default=None,
        help=(
            "Directory of reference transcript .txt files (one per WAV stem). "
            "Required for WER computation. Without this, WER column shows '—'."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size: tiny, base, small, medium, large. Default: base.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit evaluation to the first N files per mode (for quick tests).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write per-file results to a JSON file.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    for label, d in [("Clean", args.clean_dir), ("Noisy", args.noisy_dir)]:
        if not d.is_dir():
            log.error("%s directory does not exist: %s", label, d)
            sys.exit(1)

    enhanced_dirs = parse_mode_path_pairs(args.enhanced_dirs)

    # Load reference transcripts and set up Whisper
    ref_transcripts: dict[str, str] = {}
    transcriber: WhisperTranscriber | None = None
    if args.ref_transcripts is not None:
        ref_transcripts = load_ref_transcripts(args.ref_transcripts)
        if ref_transcripts:
            log.info("Loaded %d reference transcripts.", len(ref_transcripts))
            transcriber = WhisperTranscriber(args.whisper_model)
        else:
            log.warning(
                "No .txt files found in %s — WER will not be computed.",
                args.ref_transcripts,
            )

    # Build ordered mode list: Bypass first, then user-specified modes
    all_modes: dict[str, Path] = {Mode.BYPASS.value: args.noisy_dir}
    for mode, path in enhanced_dirs.items():
        all_modes[mode] = path

    # Evaluate each mode
    t0 = time.monotonic()
    results: list[ModeResult] = []
    for mode, enh_dir in all_modes.items():
        if not enh_dir.is_dir():
            log.error("Directory missing for mode '%s': %s", mode, enh_dir)
            continue
        result = evaluate_mode(
            mode, enh_dir, args.clean_dir, ref_transcripts, transcriber, args.max_files
        )
        results.append(result)

    elapsed = time.monotonic() - t0

    if not results:
        log.error("No modes evaluated.")
        sys.exit(1)

    # Print summary table
    rows = build_summary_table(results)
    print()
    print(tabulate(rows, headers="keys", tablefmt="github"))
    print()
    log.info("Evaluation completed in %.1f s.", elapsed)

    # Write JSON
    if args.output_json is not None:
        json_data: dict[str, Any] = {
            "whisper_model": args.whisper_model,
            "modes": {
                r.mode: {
                    "mean_pesq": r.mean_pesq,
                    "mean_stoi": r.mean_stoi,
                    "mean_wer": r.mean_wer,
                    "files": [
                        {
                            "name": fr.filename,
                            "pesq": fr.pesq_score,
                            "stoi": fr.stoi_score,
                            "wer": fr.wer_score,
                        }
                        for fr in r.file_results
                    ],
                }
                for r in results
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(json_data, indent=2))
        log.info("Results written to %s", args.output_json)


if __name__ == "__main__":
    main()
