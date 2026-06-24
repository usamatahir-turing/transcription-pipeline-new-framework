#!/usr/bin/env python3
"""Add short RMS vocalization segments outside model-detected speech regions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0
DEFAULT_NOISE_PERCENTILE = 20.0
DEFAULT_THRESHOLD_DB = 10.0
DEFAULT_HYSTERESIS_DB = 3.0
DEFAULT_MIN_SILENCE_MS = 40.0
DEFAULT_MIN_SHORT_EVENT_S = 0.05
DEFAULT_MAX_SHORT_EVENT_S = 1.0
DEFAULT_MIN_PEAK = 0.003
DEFAULT_MIN_MEAN_RMS = 0.0005
DEFAULT_NOISE_FLOOR_MIN = 1e-5

MODEL_SEGLST_SUFFIX = "_model.seglst.json"
OUTPUT_SUFFIX = "_model_with_rms_uncovered.seglst.json"


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"


@dataclass
class EnergyParams:
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE
    threshold_db: float = DEFAULT_THRESHOLD_DB
    hysteresis_db: float = DEFAULT_HYSTERESIS_DB
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS


def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


def _compute_frame_rms(
    wav: np.ndarray, sr: int, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if len(wav) < frame_len:
        t = np.array([len(wav) / (2.0 * sr)], dtype=np.float64)
        rms = np.array([np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-20)])
        return t, rms

    n_frames = 1 + (len(wav) - frame_len) // hop_len
    times = np.empty(n_frames, dtype=np.float64)
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = wav[start : start + frame_len].astype(np.float64)
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-20)
        times[i] = (start + frame_len * 0.5) / sr
    return times, rms


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _active_mask_hysteresis(
    rms: np.ndarray, thresh_on: float, thresh_off: float
) -> np.ndarray:
    active = np.zeros(len(rms), dtype=bool)
    in_region = False
    for i, level in enumerate(rms):
        if not in_region:
            if level >= thresh_on:
                in_region = True
                active[i] = True
        else:
            active[i] = True
            if level < thresh_off:
                in_region = False
    return active


def _estimate_noise_floor(rms: np.ndarray, percentile: float) -> float:
    """Estimate noise floor; avoid zero when many silent frames exist."""
    if len(rms) == 0:
        return DEFAULT_NOISE_FLOOR_MIN
    nonzero = rms[rms > 0]
    if len(nonzero) >= 10:
        floor = float(np.percentile(nonzero, percentile))
    else:
        floor = float(np.percentile(rms, percentile))
    return max(floor, DEFAULT_NOISE_FLOOR_MIN)


def _rms_thresholds(
    wav: np.ndarray, sr: int, params: EnergyParams
) -> tuple[float, float, float]:
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    noise_floor = _estimate_noise_floor(rms, params.noise_percentile)
    thresh_on = max(noise_floor * _db_to_linear(params.threshold_db), 1e-10)
    thresh_off = max(
        noise_floor * _db_to_linear(params.threshold_db - params.hysteresis_db),
        1e-10,
    )
    return noise_floor, thresh_on, thresh_off


def _active_regions_in_wav(
    wav: np.ndarray,
    sr: int,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
    time_offset: float = 0.0,
) -> list[tuple[float, float]]:
    """Return active (start, end) regions, optionally offset to absolute time."""
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return []

    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    if not active.any():
        return []

    frame_len = max(1, int(sr * params.frame_ms / 1000.0))
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    min_silence_frames = max(1, int(params.min_silence_ms / params.hop_ms))

    regions: list[tuple[float, float]] = []
    n = len(active)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start_idx = i
        end_idx = i
        i += 1
        while i < n:
            if active[i]:
                end_idx = i
                i += 1
                continue
            j = i
            while j < n and not active[j]:
                j += 1
            gap = j - i
            if gap < min_silence_frames and j < n:
                end_idx = j
                i = j
            else:
                break
        start_t = time_offset + (start_idx * hop_len) / sr
        end_t = time_offset + min(((end_idx * hop_len) + frame_len) / sr, len(wav) / sr)
        regions.append((start_t, end_t))
    return regions


def _merge_annotations(
    annotations: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    ann = sorted((s, e) for s, e in annotations if e > s)
    if not ann:
        return []
    merged = [ann[0]]
    for start, end in ann[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _non_speech_gaps(
    annotations: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Return intervals where model segments do not cover the timeline."""
    merged = _merge_annotations(annotations)
    if not merged:
        return [(0.0, duration)]

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _filter_by_duration(
    segments: list[tuple[float, float]],
    min_duration_s: float,
    max_duration_s: float,
) -> list[tuple[float, float]]:
    return [
        (start, end)
        for start, end in segments
        if min_duration_s <= (end - start) <= max_duration_s
    ]


def _filter_by_energy(
    wav: np.ndarray,
    sr: int,
    segments: list[tuple[float, float]],
    min_peak: float,
    min_mean_rms: float,
) -> list[tuple[float, float]]:
    """Drop segments that are too quiet or are single-sample spikes."""
    kept: list[tuple[float, float]] = []
    for start, end in segments:
        i0 = max(0, int(start * sr))
        i1 = min(len(wav), int(end * sr))
        if i1 <= i0:
            continue
        chunk = wav[i0:i1]
        peak = float(np.max(np.abs(chunk)))
        if peak < min_peak:
            continue
        mean_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if mean_rms < min_mean_rms:
            continue
        kept.append((start, end))
    return kept


def detect_short_rms_segments(
    wav_path: Path,
    annotations: list[tuple[float, float]],
    energy_params: EnergyParams | None = None,
    min_duration_s: float = DEFAULT_MIN_SHORT_EVENT_S,
    max_duration_s: float = DEFAULT_MAX_SHORT_EVENT_S,
    min_peak: float = DEFAULT_MIN_PEAK,
    min_mean_rms: float = DEFAULT_MIN_MEAN_RMS,
) -> list[tuple[float, float]]:
    """Detect short RMS bursts only outside model-detected segments."""
    params = energy_params or EnergyParams()
    wav, sr = _load_mono_wav(wav_path)
    duration = len(wav) / sr

    candidates: list[tuple[float, float]] = []
    for gap_start, gap_end in _non_speech_gaps(annotations, duration):
        if gap_end - gap_start < min_duration_s:
            continue
        i0 = int(gap_start * sr)
        i1 = min(int(gap_end * sr), len(wav))
        if i1 <= i0:
            continue
        chunk = wav[i0:i1]
        # Thresholds from this gap only (not the whole file).
        _, thresh_on, thresh_off = _rms_thresholds(chunk, sr, params)
        regions = _active_regions_in_wav(
            chunk, sr, thresh_on, thresh_off, params, time_offset=gap_start
        )
        candidates.extend(regions)

    candidates = _filter_by_duration(candidates, min_duration_s, max_duration_s)
    return _filter_by_energy(wav, sr, candidates, min_peak, min_mean_rms)


def _parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def load_seglst(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: _parse_time(item["start_time"]))


def segment_dict(
    session_id: str, speaker: str, start: float, end: float, detection_method: str
) -> dict:
    return {
        "session_id": session_id,
        "speaker": speaker,
        "start_time": f"{start:.3f}",
        "end_time": f"{end:.3f}",
        "words": "",
        "detection_method": detection_method,
    }


def model_entry(session_id: str, speaker: str, item: dict) -> dict:
    return {
        "session_id": session_id,
        "speaker": speaker,
        "start_time": f"{_parse_time(item['start_time']):.3f}",
        "end_time": f"{_parse_time(item['end_time']):.3f}",
        "words": "",
        "detection_method": str(item.get("detection_method", "model")),
    }


def speaker_from_model_path(model_path: Path) -> str:
    stem = model_path.name[: -len(MODEL_SEGLST_SUFFIX)]
    if not stem:
        raise ValueError(f"Unexpected model seglst filename: {model_path.name}")
    return stem


def output_path_for_model(model_path: Path) -> Path:
    speaker = speaker_from_model_path(model_path)
    return model_path.with_name(f"{speaker}{OUTPUT_SUFFIX}")


def wav_path_for_model(model_path: Path) -> Path:
    speaker = speaker_from_model_path(model_path)
    return model_path.with_name(f"{speaker}.wav")


def collect_jobs(input_root: Path, session: str | None) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    if session:
        session_dirs = [input_root / session]
        if not session_dirs[0].is_dir():
            raise FileNotFoundError(f"Session folder not found: {session_dirs[0]}")
    else:
        session_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    for session_dir in session_dirs:
        for model_path in sorted(session_dir.glob(f"*{MODEL_SEGLST_SUFFIX}")):
            wav_path = wav_path_for_model(model_path)
            out_path = output_path_for_model(model_path)
            jobs.append((model_path, wav_path, out_path))
    return jobs


def combine_with_rms(
    wav_path: Path, model_segments: list[dict]
) -> list[dict]:
    """Add RMS segments outside model regions; return combined seglst (no file write)."""
    session_id = wav_path.parent.name
    speaker = wav_path.stem
    annotations = [
        (_parse_time(item["start_time"]), _parse_time(item["end_time"]))
        for item in model_segments
        if _parse_time(item["end_time"]) > _parse_time(item["start_time"])
    ]
    rms_segments = detect_short_rms_segments(wav_path, annotations)
    combined = [model_entry(session_id, speaker, item) for item in model_segments]
    for start, end in rms_segments:
        combined.append(segment_dict(session_id, speaker, start, end, "rms"))
    combined.sort(key=lambda item: _parse_time(item["start_time"]))
    return combined


def process_pair(model_path: Path, wav_path: Path, output_path: Path) -> tuple[int, int]:
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing WAV for {model_path.name}: {wav_path}")

    model_segments = load_seglst(model_path)
    combined = combine_with_rms(wav_path, model_segments)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)
        fh.write("\n")

    rms_count = sum(1 for item in combined if item.get("detection_method") == "rms")
    return len(model_segments), rms_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add short RMS segments outside model speech regions to seglst files."
    )
    default_input = default_input_root()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
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

    jobs = collect_jobs(input_root, args.session)
    if not jobs:
        log.warning("No *_model.seglst.json files found under %s", input_root)
        return 0

    to_run: list[tuple[Path, Path, Path]] = []
    skipped = 0
    for model_path, wav_path, out_path in jobs:
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        if not wav_path.is_file():
            log.error("Missing WAV for %s: %s", model_path.name, wav_path.name)
            return 1
        to_run.append((model_path, wav_path, out_path))

    log.info(
        "Found %d model seglst file(s): %d to process, %d skipped",
        len(jobs),
        len(to_run),
        skipped,
    )
    if not to_run:
        return 0

    t_total = time.time()
    total_model = 0
    total_rms = 0
    for model_path, wav_path, out_path in to_run:
        t0 = time.time()
        model_count, rms_count = process_pair(model_path, wav_path, out_path)
        total_model += model_count
        total_rms += rms_count
        log.info(
            "%s -> %s (%d model + %d rms = %d segments, %.1fs)",
            model_path.name,
            out_path.name,
            model_count,
            rms_count,
            model_count + rms_count,
            time.time() - t0,
        )

    log.info(
        "Done: %d file(s), %d model + %d rms segment(s) in %.1fs",
        len(to_run),
        total_model,
        total_rms,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())