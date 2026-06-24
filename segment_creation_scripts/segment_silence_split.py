#!/usr/bin/env python3
"""Split model segments at interior silence gaps longer than 300 ms."""
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

INPUT_SUFFIX = "_model_with_rms_uncovered.seglst.json"
OUTPUT_SUFFIX = "_model_with_rms_uncovered_split_silence.seglst.json"
MODEL_METHOD = "model"
SPLIT_METHOD = "model_and_silence_split"


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"

DEFAULT_MAX_SILENCE_S = 0.2
DEFAULT_SPLIT_MARGIN_S = 0.1

DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0
DEFAULT_NOISE_PERCENTILE = 15.0
DEFAULT_THRESHOLD_DB = 12.0
DEFAULT_HYSTERESIS_DB = 3.0


@dataclass
class EnergyParams:
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE
    threshold_db: float = DEFAULT_THRESHOLD_DB
    hysteresis_db: float = DEFAULT_HYSTERESIS_DB


@dataclass
class SilenceGap:
    start: float
    end: float


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


def _file_noise_floor(
    wav: np.ndarray, sr: int, params: EnergyParams
) -> tuple[float, float, float]:
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    noise_floor = float(np.percentile(rms, params.noise_percentile))
    thresh_on = max(noise_floor * _db_to_linear(params.threshold_db), 1e-10)
    thresh_off = max(
        noise_floor * _db_to_linear(params.threshold_db - params.hysteresis_db),
        1e-10,
    )
    return noise_floor, thresh_on, thresh_off


def _local_active_mask(
    wav: np.ndarray,
    sr: int,
    seg_start: float,
    seg_end: float,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
) -> tuple[np.ndarray, int, int]:
    i0 = int(seg_start * sr)
    i1 = int(seg_end * sr)
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    if i1 <= i0:
        return np.zeros(0, dtype=bool), i0, hop_len

    _, rms = _compute_frame_rms(wav[i0:i1], sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return np.zeros(0, dtype=bool), i0, hop_len
    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    return active, i0, hop_len


def _silence_gaps_from_mask(
    active: np.ndarray,
    seg_start: float,
    seg_end: float,
    sr: int,
    hop_len: int,
    max_silence_s: float,
    ignore_edges: bool,
) -> list[SilenceGap]:
    if len(active) == 0 or not active.any():
        return []

    active_idx = np.where(active)[0]
    first_active = int(active_idx[0])
    last_active = int(active_idx[-1])

    gaps: list[SilenceGap] = []
    i = first_active + 1
    while i <= last_active:
        if active[i]:
            i += 1
            continue
        j = i
        while j <= last_active and not active[j]:
            j += 1
        gap_start = seg_start + (i * hop_len) / sr
        gap_end = seg_start + (j * hop_len) / sr
        if (gap_end - gap_start) > max_silence_s:
            gaps.append(SilenceGap(gap_start, gap_end))
        i = j + 1

    if not ignore_edges:
        n = len(active)
        if first_active > 0:
            edge = SilenceGap(seg_start, seg_start + (first_active * hop_len) / sr)
            if edge.end - edge.start > max_silence_s:
                gaps.insert(0, edge)
        if last_active < n - 1:
            edge = SilenceGap(
                seg_start + ((last_active + 1) * hop_len) / sr, seg_end
            )
            if edge.end - edge.start > max_silence_s:
                gaps.append(edge)

    return gaps


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


def split_segment_at_gaps(
    seg_start: float,
    seg_end: float,
    gaps: list[SilenceGap],
    margin_s: float,
) -> list[tuple[float, float]]:
    """Split a segment at silence gap centers with margin on each side."""
    if not gaps:
        return [(seg_start, seg_end)]

    cuts: list[tuple[float, float]] = []
    for gap in sorted(gaps, key=lambda g: g.start):
        center = (gap.start + gap.end) / 2.0
        cuts.append((center - margin_s, center + margin_s))

    pieces: list[tuple[float, float]] = []
    cursor = seg_start
    for left_end, right_start in cuts:
        if left_end > cursor:
            pieces.append((cursor, left_end))
        cursor = max(cursor, right_start)
    if cursor < seg_end:
        pieces.append((cursor, seg_end))

    return [(start, end) for start, end in pieces if end > start]


def interior_silence_gaps(
    wav: np.ndarray,
    sr: int,
    seg_start: float,
    seg_end: float,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
    max_silence_s: float,
) -> list[SilenceGap]:
    active, _, hop_len = _local_active_mask(
        wav, sr, seg_start, seg_end, thresh_on, thresh_off, params
    )
    return _silence_gaps_from_mask(
        active,
        seg_start,
        seg_end,
        sr,
        hop_len,
        max_silence_s,
        ignore_edges=True,
    )


def segment_dict(
    session_id: str,
    speaker: str,
    start: float,
    end: float,
    detection_method: str,
    words: str = "",
) -> dict:
    return {
        "session_id": session_id,
        "speaker": speaker,
        "start_time": f"{start:.3f}",
        "end_time": f"{end:.3f}",
        "words": words,
        "detection_method": detection_method,
    }


def speaker_from_input_path(input_path: Path) -> str:
    stem = input_path.name[: -len(INPUT_SUFFIX)]
    if not stem:
        raise ValueError(f"Unexpected input seglst filename: {input_path.name}")
    return stem


def output_path_for_input(input_path: Path) -> Path:
    speaker = speaker_from_input_path(input_path)
    return input_path.with_name(f"{speaker}{OUTPUT_SUFFIX}")


def wav_path_for_input(input_path: Path) -> Path:
    speaker = speaker_from_input_path(input_path)
    return input_path.with_name(f"{speaker}.wav")


def collect_jobs(input_root: Path, session: str | None) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    if session:
        session_dirs = [input_root / session]
        if not session_dirs[0].is_dir():
            raise FileNotFoundError(f"Session folder not found: {session_dirs[0]}")
    else:
        session_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    for session_dir in session_dirs:
        for input_path in sorted(session_dir.glob(f"*{INPUT_SUFFIX}")):
            wav_path = wav_path_for_input(input_path)
            out_path = output_path_for_input(input_path)
            jobs.append((input_path, wav_path, out_path))
    return jobs


def split_seglst(
    wav_path: Path,
    segments: list[dict],
    max_silence_s: float = DEFAULT_MAX_SILENCE_S,
    split_margin_s: float = DEFAULT_SPLIT_MARGIN_S,
    energy_params: EnergyParams | None = None,
) -> tuple[list[dict], int, int, int]:
    """Split model segments at interior silence; return (seglst, model_count, split_count, unchanged)."""
    params = energy_params or EnergyParams()
    session_id = wav_path.parent.name
    speaker = wav_path.stem
    wav, sr = _load_mono_wav(wav_path)
    _, thresh_on, thresh_off = _file_noise_floor(wav, sr, params)

    output: list[dict] = []
    model_count = 0
    split_count = 0
    unchanged_model = 0

    for item in segments:
        method = str(item.get("detection_method", ""))
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        words = str(item.get("words", ""))

        if method != MODEL_METHOD:
            output.append(
                segment_dict(session_id, speaker, start, end, method, words)
            )
            continue

        model_count += 1
        gaps = interior_silence_gaps(
            wav, sr, start, end, thresh_on, thresh_off, params, max_silence_s
        )
        pieces = split_segment_at_gaps(start, end, gaps, split_margin_s)

        if len(pieces) == 1 and abs(pieces[0][0] - start) < 1e-9 and abs(pieces[0][1] - end) < 1e-9:
            output.append(
                segment_dict(session_id, speaker, start, end, MODEL_METHOD, words)
            )
            unchanged_model += 1
            continue

        split_count += 1
        for piece_start, piece_end in pieces:
            output.append(
                segment_dict(
                    session_id,
                    speaker,
                    piece_start,
                    piece_end,
                    SPLIT_METHOD,
                    words,
                )
            )

    output.sort(key=lambda item: _parse_time(item["start_time"]))
    return output, model_count, split_count, unchanged_model


def process_pair(
    input_path: Path,
    wav_path: Path,
    output_path: Path,
    max_silence_s: float,
    split_margin_s: float,
    energy_params: EnergyParams | None = None,
) -> tuple[int, int, int]:
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing WAV for {input_path.name}: {wav_path}")

    segments = load_seglst(input_path)
    output, model_count, split_count, unchanged_model = split_seglst(
        wav_path, segments, max_silence_s, split_margin_s, energy_params
    )

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")

    return model_count, split_count, unchanged_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split model segments at interior silence gaps in "
            "*_model_with_rms_uncovered.seglst.json files."
        )
    )
    default_input = default_input_root()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-silence",
        type=float,
        default=DEFAULT_MAX_SILENCE_S,
        help="Interior silence longer than this triggers a split (default: 0.3 s)",
    )
    parser.add_argument(
        "--split-margin",
        type=float,
        default=DEFAULT_SPLIT_MARGIN_S,
        help="Seconds before/after silence center for split boundaries (default: 0.1)",
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

    jobs = collect_jobs(input_root, args.session)
    if not jobs:
        log.warning("No %s files found under %s", INPUT_SUFFIX, input_root)
        return 0

    to_run: list[tuple[Path, Path, Path]] = []
    skipped = 0
    for input_path, wav_path, out_path in jobs:
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        if not wav_path.is_file():
            log.error("Missing WAV for %s: %s", input_path.name, wav_path.name)
            return 1
        to_run.append((input_path, wav_path, out_path))

    log.info(
        "Found %d input seglst file(s): %d to process, %d skipped",
        len(jobs),
        len(to_run),
        skipped,
    )
    if not to_run:
        return 0

    t_total = time.time()
    total_model = 0
    total_split = 0
    total_unchanged = 0
    for input_path, wav_path, out_path in to_run:
        t0 = time.time()
        model_count, split_count, unchanged = process_pair(
            input_path,
            wav_path,
            out_path,
            args.max_silence,
            args.split_margin,
        )
        total_model += model_count
        total_split += split_count
        total_unchanged += unchanged
        log.info(
            "%s -> %s (%d model, %d split, %d unchanged model, %.1fs)",
            input_path.name,
            out_path.name,
            model_count,
            split_count,
            unchanged,
            time.time() - t0,
        )

    log.info(
        "Done: %d file(s), %d model segment(s), %d split, %d unchanged in %.1fs",
        len(to_run),
        total_model,
        total_split,
        total_unchanged,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
