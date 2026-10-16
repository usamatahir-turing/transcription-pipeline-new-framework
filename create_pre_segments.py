#!/usr/bin/env python3
"""Segmentation + boundary fix without ASR.

Writes {speaker}.seglst.json (pre-trim merge, empty words) and
{speaker}_final.seglst.json (same boundary trim as create_segments.py full run).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from finalization_scripts.segment_boundary_fix import (
    DEFAULT_MIN_TRIM_S,
    DEFAULT_REF_MS,
    EdgeTrimParams,
    fix_segment_boundaries,
    load_seglst,
)

from create_segments import (
    default_input_root,
    intermediate_paths,
    maybe_write_seglst,
    normalize_speaker_id,
    resolve_wav_jobs,
    run_pipeline,
    write_seglst,
)
from segment_creation_scripts.speech_detection import (
    _get_torch_device,
    collect_wav_jobs,
)

log = logging.getLogger(__name__)


def plan_pre_stages(
    paths: dict[str, Path],
    *,
    overwrite: bool,
) -> tuple[bool, bool, bool]:
    """Return (skip_all, do_segmentation, do_boundary_fix)."""
    merged_path = paths["final"]
    boundary_final_path = paths["boundary_final"]

    if not overwrite and boundary_final_path.is_file():
        return True, False, False

    if overwrite:
        return False, True, True

    do_segmentation = not merged_path.is_file()
    do_boundary_fix = not boundary_final_path.is_file()
    return False, do_segmentation, do_boundary_fix


def process_wav(
    wav_path: Path,
    model: str,
    *,
    overwrite: bool,
    write_intermediates: bool,
    overwrite_intermediates: bool,
    min_trim_s: float,
    ref_ms: float,
) -> tuple[int, int] | None:
    paths = intermediate_paths(wav_path, model)
    merged_path = paths["final"]
    boundary_final_path = paths["boundary_final"]

    skip_all, do_segmentation, do_boundary_fix = plan_pre_stages(
        paths, overwrite=overwrite
    )
    if skip_all:
        log.info("Skipping (exists): %s", boundary_final_path.name)
        return None

    stage_names = [
        name
        for name, enabled in (
            ("segmentation", do_segmentation),
            ("boundary fix", do_boundary_fix),
        )
        if enabled
    ]
    log.info("Running %s for %s", " -> ".join(stage_names), wav_path.name)

    merged: list[dict]
    if do_segmentation:
        t0 = time.time()
        speech, rms_uncovered, split_silence, merged = run_pipeline(wav_path, model)

        write_inter = write_intermediates or overwrite_intermediates
        maybe_write_seglst(
            paths["speech"],
            speech,
            enabled=write_inter,
            overwrite=overwrite_intermediates,
        )
        maybe_write_seglst(
            paths["rms_uncovered"],
            rms_uncovered,
            enabled=write_inter,
            overwrite=overwrite_intermediates,
        )
        maybe_write_seglst(
            paths["split_silence"],
            split_silence,
            enabled=write_inter,
            overwrite=overwrite_intermediates,
        )
        maybe_write_seglst(
            paths["merged"],
            merged,
            enabled=write_inter,
            overwrite=overwrite_intermediates,
        )

        write_seglst(merged_path, merged)
        log.info(
            "%s -> %s (%d segments, %.1fs)",
            wav_path.name,
            merged_path.name,
            len(merged),
            time.time() - t0,
        )
    else:
        merged = load_seglst(merged_path)
        log.info("Loaded existing %s (%d segments)", merged_path.name, len(merged))

    boundary_fixes = 0
    if do_boundary_fix:
        t_fix = time.time()
        trim_params = EdgeTrimParams(ref_ms=ref_ms, min_trim_s=min_trim_s)
        final_segments, onset_fixes, offset_fixes = fix_segment_boundaries(
            wav_path,
            merged,
            trim_params=trim_params,
        )
        write_seglst(boundary_final_path, final_segments)
        boundary_fixes = onset_fixes + offset_fixes
        log.info(
            "Finalized %s -> %s (%d onset + %d offset fixes, %.1fs)",
            merged_path.name,
            boundary_final_path.name,
            onset_fixes,
            offset_fixes,
            time.time() - t_fix,
        )

    return len(merged), boundary_fixes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run segmentation + boundary fix (no ASR). Writes "
            "{speaker}.seglst.json and {speaker}_final.seglst.json with the same "
            "trimmed boundaries as create_segments.py after full finalization."
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
    parser.add_argument(
        "--speaker",
        type=str,
        nargs="+",
        default=None,
        metavar="SPEAKER",
        help=(
            "Speaker WAV stem(s), e.g. felipe.m1@turing.com "
            "(requires --session; processes only matching WAVs)"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["model", "silero", "sortformer"],
        default="model",
        help="Speech detection mode (default: model = Silero union Sortformer)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run segmentation and boundary fix, ignoring existing "
            "{speaker}.seglst.json and {speaker}_final.seglst.json"
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Also write intermediate seglst files (skip existing unless --overwrite-intermediates)",
    )
    parser.add_argument(
        "--overwrite-intermediates",
        action="store_true",
        help="Write intermediate seglst files and overwrite any that already exist",
    )
    parser.add_argument(
        "--min-trim",
        type=float,
        default=DEFAULT_MIN_TRIM_S,
        help="Apply boundary fix only when trim exceeds this many seconds (default: 0.02)",
    )
    parser.add_argument(
        "--ref-ms",
        type=float,
        default=DEFAULT_REF_MS,
        help="Edge reference window in ms for baseline RMS (default: 150)",
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

    if args.speaker and not args.session:
        parser.error("--speaker requires at least one --session")

    if args.speaker:
        speakers = [normalize_speaker_id(name) for name in args.speaker]
        wav_jobs = resolve_wav_jobs(input_root, list(args.session), speakers)
    else:
        wav_jobs = collect_wav_jobs(input_root, args.session)
    if not wav_jobs:
        log.warning("No WAV files found under %s", input_root)
        return 0

    log.info("Using device: %s", _get_torch_device())

    t_total = time.time()
    processed = 0
    skipped = 0
    total_segments = 0
    total_boundary_fixes = 0

    for wav_path in wav_jobs:
        result = process_wav(
            wav_path,
            args.model,
            overwrite=args.overwrite,
            write_intermediates=args.keep_intermediates,
            overwrite_intermediates=args.overwrite_intermediates,
            min_trim_s=args.min_trim,
            ref_ms=args.ref_ms,
        )
        if result is None:
            skipped += 1
        else:
            seg_count, fix_count = result
            processed += 1
            total_segments += seg_count
            total_boundary_fixes += fix_count

    log.info(
        "Done: mode=%s | %d processed, %d skipped, %d segment(s), "
        "%d boundary fix(es) in %.1fs",
        args.model,
        processed,
        skipped,
        total_segments,
        total_boundary_fixes,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
