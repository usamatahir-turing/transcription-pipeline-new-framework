#!/usr/bin/env python3
"""Trim segment edge padding using a file-wide edge RMS baseline."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

INPUT_SUFFIX = "_qwen3.seglst.json"
OUTPUT_SUFFIX = "_final.seglst.json"

DEFAULT_REF_MS = 150.0
DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0
DEFAULT_EDGE_MARGIN = 1.8
DEFAULT_MIN_TRIM_S = 0.02
DEFAULT_MAX_REF_FRACTION = 0.25
DEFAULT_REF_PERCENTILE = 25.0
DEFAULT_SPEECH_SKIP_RATIO = 6.0
DEFAULT_NOISE_PERCENTILE = 15.0


@dataclass
class EdgeTrimParams:
    ref_ms: float = DEFAULT_REF_MS
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    edge_margin: float = DEFAULT_EDGE_MARGIN
    min_trim_s: float = DEFAULT_MIN_TRIM_S
    max_ref_fraction: float = DEFAULT_MAX_REF_FRACTION
    ref_percentile: float = DEFAULT_REF_PERCENTILE
    speech_skip_ratio: float = DEFAULT_SPEECH_SKIP_RATIO
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE


@dataclass(frozen=True)
class GlobalBaselines:
    leading: float
    trailing: float
    noise_floor: float


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"


def input_path_for_wav(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{INPUT_SUFFIX}")


def output_path_for_wav(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{OUTPUT_SUFFIX}")


def _parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _format_time(seconds: float) -> str:
    return f"{seconds:.3f}"


def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


def _window_rms_series(
    wav: np.ndarray, sr: int, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, int, int]:
    """Return per-window RMS; window i starts at sample i * hop_len."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if len(wav) < frame_len:
        rms = np.array(
            [float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-20))],
            dtype=np.float64,
        )
        return rms, frame_len, hop_len

    n_frames = 1 + (len(wav) - frame_len) // hop_len
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = wav[start : start + frame_len].astype(np.float64)
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-20)
    return rms, frame_len, hop_len


def _ref_samples(seg_samples: int, sr: int, params: EdgeTrimParams) -> int:
    ref = int(sr * params.ref_ms / 1000.0)
    cap = max(1, int(seg_samples * params.max_ref_fraction))
    return max(1, min(ref, cap, seg_samples))


def _ref_region_levels(
    rms: np.ndarray,
    frame_len: int,
    hop_len: int,
    ref_samples: int,
    *,
    from_end: bool,
) -> list[float]:
    if len(rms) == 0:
        return []

    ref_start_sample = (
        0
        if not from_end
        else max(0, (len(rms) - 1) * hop_len + frame_len - ref_samples)
    )
    ref_end_sample = ref_samples if not from_end else len(rms) * hop_len + frame_len

    levels: list[float] = []
    for i, level in enumerate(rms):
        win_start = i * hop_len
        win_end = win_start + frame_len
        win_mid = (win_start + win_end) // 2
        if from_end:
            if win_mid >= ref_start_sample:
                levels.append(float(level))
        elif win_mid < ref_end_sample:
            levels.append(float(level))
    return levels


def _file_noise_floor(
    wav: np.ndarray, sr: int, params: EdgeTrimParams
) -> float:
    rms, _, _ = _window_rms_series(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return 1e-10
    return max(float(np.percentile(rms, params.noise_percentile)), 1e-10)


def _pool_percentile(levels: list[float], percentile: float, fallback: float) -> float:
    if not levels:
        return fallback
    return max(float(np.percentile(levels, percentile)), 1e-10)


def _compute_global_baselines(
    wav: np.ndarray,
    sr: int,
    segments: list[dict],
    params: EdgeTrimParams,
) -> GlobalBaselines:
    noise_floor = _file_noise_floor(wav, sr, params)
    leading_levels: list[float] = []
    trailing_levels: list[float] = []

    for item in segments:
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        if end <= start:
            continue

        i0 = int(start * sr)
        i1 = int(end * sr)
        if i1 <= i0:
            continue

        chunk = wav[i0:i1]
        rms, frame_len, hop_len = _window_rms_series(
            chunk, sr, params.frame_ms, params.hop_ms
        )
        ref_samples = _ref_samples(len(chunk), sr, params)
        if ref_samples < frame_len:
            continue

        leading_levels.extend(
            _ref_region_levels(
                rms, frame_len, hop_len, ref_samples, from_end=False
            )
        )
        trailing_levels.extend(
            _ref_region_levels(
                rms, frame_len, hop_len, ref_samples, from_end=True
            )
        )

    leading = _pool_percentile(leading_levels, params.ref_percentile, noise_floor)
    trailing = _pool_percentile(trailing_levels, params.ref_percentile, noise_floor)
    return GlobalBaselines(leading=leading, trailing=trailing, noise_floor=noise_floor)


def _edge_threshold(baseline: float, baselines: GlobalBaselines, margin: float) -> float:
    return max(baseline * margin, baselines.noise_floor * 2.0)


def _edge_looks_like_padding(
    rms: np.ndarray,
    frame_len: int,
    hop_len: int,
    ref_samples: int,
    *,
    from_end: bool,
    global_baseline: float,
    speech_skip_ratio: float,
) -> bool:
    """Skip trimming when the local edge is dominated by speech, not padding."""
    levels = _ref_region_levels(
        rms, frame_len, hop_len, ref_samples, from_end=from_end
    )
    if not levels:
        return False
    local = float(np.median(levels))
    return local <= global_baseline * speech_skip_ratio


def _time_at_window_start(i0: int, index: int, hop_len: int, sr: int) -> float:
    return (i0 + index * hop_len) / sr


def _time_at_window_end(
    i0: int, index: int, hop_len: int, frame_len: int, sr: int, cap_sample: int
) -> float:
    end_sample = min(i0 + index * hop_len + frame_len, cap_sample)
    return end_sample / sr


def _trim_leading_edge(
    wav: np.ndarray,
    i0: int,
    sr: int,
    seg_start: float,
    params: EdgeTrimParams,
    baselines: GlobalBaselines,
) -> float:
    rms, frame_len, hop_len = _window_rms_series(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return seg_start

    ref_samples = _ref_samples(len(wav), sr, params)
    if ref_samples < frame_len:
        return seg_start

    if not _edge_looks_like_padding(
        rms,
        frame_len,
        hop_len,
        ref_samples,
        from_end=False,
        global_baseline=baselines.leading,
        speech_skip_ratio=params.speech_skip_ratio,
    ):
        return seg_start

    threshold = _edge_threshold(baselines.leading, baselines, params.edge_margin)
    for index, level in enumerate(rms):
        if level > threshold:
            return _time_at_window_start(i0, index, hop_len, sr)
    return seg_start


def _trim_trailing_edge(
    wav: np.ndarray,
    i0: int,
    i1: int,
    sr: int,
    seg_end: float,
    params: EdgeTrimParams,
    baselines: GlobalBaselines,
) -> float:
    rms, frame_len, hop_len = _window_rms_series(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return seg_end

    ref_samples = _ref_samples(len(wav), sr, params)
    if ref_samples < frame_len:
        return seg_end

    if not _edge_looks_like_padding(
        rms,
        frame_len,
        hop_len,
        ref_samples,
        from_end=True,
        global_baseline=baselines.trailing,
        speech_skip_ratio=params.speech_skip_ratio,
    ):
        return seg_end

    threshold = _edge_threshold(baselines.trailing, baselines, params.edge_margin)
    for index in range(len(rms) - 1, -1, -1):
        if rms[index] > threshold:
            return _time_at_window_end(i0, index, hop_len, frame_len, sr, i1)
    return seg_end


def fix_segment_boundaries(
    wav_path: Path,
    segments: list[dict],
    *,
    trim_params: EdgeTrimParams | None = None,
) -> tuple[list[dict], int, int]:
    """Trim edge padding using file-wide edge RMS baselines."""
    params = trim_params or EdgeTrimParams()
    wav, sr = _load_mono_wav(wav_path)
    baselines = _compute_global_baselines(wav, sr, segments, params)

    fixed: list[dict] = []
    onset_fixes = 0
    offset_fixes = 0

    for item in segments:
        row = dict(item)
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        new_start = start
        new_end = end

        if end > start:
            i0 = int(start * sr)
            i1 = int(end * sr)
            if i1 > i0:
                chunk = wav[i0:i1]
                trimmed_start = _trim_leading_edge(
                    chunk, i0, sr, start, params, baselines
                )
                trimmed_end = _trim_trailing_edge(
                    chunk, i0, i1, sr, end, params, baselines
                )

                if trimmed_start - start >= params.min_trim_s:
                    new_start = trimmed_start
                    onset_fixes += 1
                if end - trimmed_end >= params.min_trim_s:
                    new_end = trimmed_end
                    offset_fixes += 1

        row["start_time"] = _format_time(new_start)
        row["end_time"] = _format_time(new_end)
        fixed.append(row)

    return fixed, onset_fixes, offset_fixes


def load_seglst(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: _parse_time(item["start_time"]))


def write_seglst(path: Path, segments: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def collect_jobs(
    input_root: Path,
    session: str | Sequence[str] | None,
) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    if session:
        sessions = [session] if isinstance(session, str) else list(session)
        session_dirs = []
        for name in sessions:
            session_dir = input_root / name
            if not session_dir.is_dir():
                raise FileNotFoundError(f"Session folder not found: {session_dir}")
            session_dirs.append(session_dir)
    else:
        session_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    for session_dir in session_dirs:
        for wav_path in sorted(session_dir.glob("*.wav")):
            in_path = input_path_for_wav(wav_path)
            out_path = output_path_for_wav(wav_path)
            jobs.append((wav_path, in_path, out_path))
    return jobs


def process_speaker(
    wav_path: Path,
    seglst_path: Path,
    out_path: Path,
    *,
    trim_params: EdgeTrimParams,
) -> tuple[int, int, int]:
    segments = load_seglst(seglst_path)
    fixed, onset_fixes, offset_fixes = fix_segment_boundaries(
        wav_path,
        segments,
        trim_params=trim_params,
    )
    write_seglst(out_path, fixed)
    return len(segments), onset_fixes, offset_fixes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trim edge padding in {speaker}_qwen3.seglst.json using file-wide "
            "edge RMS baselines; write {speaker}_final.seglst.json."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input_root())
    parser.add_argument(
        "--session",
        type=str,
        nargs="+",
        default=None,
        metavar="SESSION",
        help="One or more session folders under --input (default: all sessions)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ref-ms",
        type=float,
        default=DEFAULT_REF_MS,
        help="Edge reference window in ms for baseline pool (default: 150)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=DEFAULT_FRAME_MS,
        help="Sliding analysis window in ms (default: 20)",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=DEFAULT_HOP_MS,
        help="Sliding window hop in ms (default: 10)",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=DEFAULT_EDGE_MARGIN,
        help="Speech threshold = baseline * margin (default: 1.3)",
    )
    parser.add_argument(
        "--min-trim",
        type=float,
        default=DEFAULT_MIN_TRIM_S,
        help="Apply a fix only when trim exceeds this many seconds (default: 0.02)",
    )
    parser.add_argument(
        "--ref-percentile",
        type=float,
        default=DEFAULT_REF_PERCENTILE,
        help="Percentile of pooled edge-window RMS for global baseline (default: 25)",
    )
    parser.add_argument(
        "--speech-skip-ratio",
        type=float,
        default=DEFAULT_SPEECH_SKIP_RATIO,
        help=(
            "Skip trimming an edge when local ref median exceeds baseline * ratio "
            "(default: 6)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    input_root = args.input.resolve()
    if not input_root.is_dir():
        log.error("Input directory does not exist: %s", input_root)
        return 1

    trim_params = EdgeTrimParams(
        ref_ms=args.ref_ms,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        edge_margin=args.edge_margin,
        min_trim_s=args.min_trim,
        ref_percentile=args.ref_percentile,
        speech_skip_ratio=args.speech_skip_ratio,
    )

    jobs = collect_jobs(input_root, args.session)
    if not jobs:
        log.warning("No WAV files found under %s", input_root)
        return 0

    to_run: list[tuple[Path, Path, Path]] = []
    skipped = 0
    missing_input = 0
    for wav_path, in_path, out_path in jobs:
        if not in_path.is_file():
            log.warning("Skipping %s: missing %s", wav_path.name, in_path.name)
            missing_input += 1
            continue
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        to_run.append((wav_path, in_path, out_path))

    log.info(
        "Found %d speaker(s): %d to fix, %d skipped, %d missing qwen3 seglst",
        len(jobs),
        len(to_run),
        skipped,
        missing_input,
    )
    if not to_run:
        return 0

    t_total = time.time()
    total_segments = 0
    total_onset = 0
    total_offset = 0
    for n, (wav_path, in_path, out_path) in enumerate(to_run, start=1):
        t0 = time.time()
        log.info("[%d/%d] %s", n, len(to_run), wav_path.name)
        seg_count, onset_fixes, offset_fixes = process_speaker(
            wav_path,
            in_path,
            out_path,
            trim_params=trim_params,
        )
        total_segments += seg_count
        total_onset += onset_fixes
        total_offset += offset_fixes
        log.info(
            "Wrote %s (%d segments, %d onset + %d offset fixes, %.1fs)",
            out_path.name,
            seg_count,
            onset_fixes,
            offset_fixes,
            time.time() - t0,
        )

    log.info(
        "Done: %d file(s), %d segment(s), %d onset + %d offset fixes in %.1fs",
        len(to_run),
        total_segments,
        total_onset,
        total_offset,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
