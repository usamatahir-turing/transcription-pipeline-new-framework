#!/usr/bin/env python3
"""Merge adjacent seglst segments that touch exactly at a shared boundary."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

INPUT_SUFFIX = "_model_with_rms_uncovered_split_silence.seglst.json"
OUTPUT_SUFFIX = "_model_with_rms_uncovered_split_silence_merged.seglst.json"


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"


def _parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def load_seglst(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: (_parse_time(item["start_time"]), item["end_time"]))


def merge_touching_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Merge segments where end_time exactly equals the next start_time (string match)."""
    if not segments:
        return [], 0

    ordered = sorted(
        segments, key=lambda item: (_parse_time(item["start_time"]), item["end_time"])
    )
    merged: list[dict] = []
    current = dict(ordered[0])
    merge_ops = 0

    for nxt in ordered[1:]:
        if current["end_time"] == nxt["start_time"]:
            current["end_time"] = nxt["end_time"]
            current["detection_method"] = (
                f"{current['detection_method']}_merged_{nxt['detection_method']}"
            )
            merge_ops += 1
            continue
        merged.append(current)
        current = dict(nxt)

    merged.append(current)
    return merged, merge_ops


def speaker_from_input_path(input_path: Path) -> str:
    stem = input_path.name[: -len(INPUT_SUFFIX)]
    if not stem:
        raise ValueError(f"Unexpected input seglst filename: {input_path.name}")
    return stem


def output_path_for_input(input_path: Path) -> Path:
    speaker = speaker_from_input_path(input_path)
    return input_path.with_name(f"{speaker}{OUTPUT_SUFFIX}")


def collect_jobs(input_root: Path, session: str | None) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    if session:
        session_dirs = [input_root / session]
        if not session_dirs[0].is_dir():
            raise FileNotFoundError(f"Session folder not found: {session_dirs[0]}")
    else:
        session_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    for session_dir in session_dirs:
        for input_path in sorted(session_dir.glob(f"*{INPUT_SUFFIX}")):
            jobs.append((input_path, output_path_for_input(input_path)))
    return jobs


def apply_touch_merge(segments: list[dict]) -> tuple[list[dict], int]:
    """Merge touching segments; return (seglst, merge_op_count)."""
    return merge_touching_segments(segments)


def process_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    segments = load_seglst(input_path)
    merged, merge_ops = apply_touch_merge(segments)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
        fh.write("\n")

    return len(segments), len(merged), merge_ops


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge touching segments in "
            "*_model_with_rms_uncovered_split_silence.seglst.json files."
        )
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
        log.warning("No %s files found under %s", INPUT_SUFFIX, input_root)
        return 0

    to_run: list[tuple[Path, Path]] = []
    skipped = 0
    for input_path, out_path in jobs:
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        to_run.append((input_path, out_path))

    log.info(
        "Found %d input seglst file(s): %d to process, %d skipped",
        len(jobs),
        len(to_run),
        skipped,
    )
    if not to_run:
        return 0

    t_total = time.time()
    total_in = 0
    total_out = 0
    total_ops = 0
    for input_path, out_path in to_run:
        t0 = time.time()
        in_count, out_count, merge_ops = process_file(input_path, out_path)
        total_in += in_count
        total_out += out_count
        total_ops += merge_ops
        log.info(
            "%s -> %s (%d -> %d segments, %d merge(s), %.1fs)",
            input_path.name,
            out_path.name,
            in_count,
            out_count,
            merge_ops,
            time.time() - t0,
        )

    log.info(
        "Done: %d file(s), %d -> %d segment(s), %d merge op(s) in %.1fs",
        len(to_run),
        total_in,
        total_out,
        total_ops,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
