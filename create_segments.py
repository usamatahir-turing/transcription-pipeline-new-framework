#!/usr/bin/env python3
"""Run segment creation + Qwen3 ASR + boundary fix; write .seglst.json, _qwen3, and _final outputs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from finalization_scripts.segment_boundary_fix import (
    DEFAULT_MIN_TRIM_S,
    DEFAULT_REF_MS,
    EdgeTrimParams,
    OUTPUT_SUFFIX as BOUNDARY_FINAL_SUFFIX,
    fix_segment_boundaries,
    load_seglst,
)

from segment_creation_scripts.segment_silence_split import (
    DEFAULT_MAX_SILENCE_S,
    DEFAULT_SPLIT_MARGIN_S,
    OUTPUT_SUFFIX as SPLIT_SILENCE_SUFFIX,
    split_seglst,
)
from segment_creation_scripts.segment_touch_merge import (
    OUTPUT_SUFFIX as MERGED_SUFFIX,
    apply_touch_merge,
)
from segment_creation_scripts.speech_detection import (
    _get_torch_device,
    build_seglst,
    collect_wav_jobs,
    seglst_output_path,
)
from segment_creation_scripts.uncovered_segment_detection_rms import (
    OUTPUT_SUFFIX as RMS_UNCOVERED_SUFFIX,
    combine_with_rms,
)
from transcription_scripts.qwen3_asr_gen import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    OUTPUT_SUFFIX as QWEN3_SUFFIX,
    build_model,
    resolve_language,
    should_transcribe,
    transcribe_seglst_segments,
)

log = logging.getLogger(__name__)

FINAL_SUFFIX = ".seglst.json"


def default_input_root() -> Path:
    return Path(__file__).resolve().parent / "input_audio_files"


def final_seglst_path(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{FINAL_SUFFIX}")


def qwen3_seglst_path(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{QWEN3_SUFFIX}")


def boundary_final_seglst_path(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{BOUNDARY_FINAL_SUFFIX}")


def intermediate_paths(wav_path: Path, model: str) -> dict[str, Path]:
    speaker = wav_path.stem
    parent = wav_path.parent
    return {
        "speech": seglst_output_path(wav_path, model),
        "rms_uncovered": parent / f"{speaker}{RMS_UNCOVERED_SUFFIX}",
        "split_silence": parent / f"{speaker}{SPLIT_SILENCE_SUFFIX}",
        "merged": parent / f"{speaker}{MERGED_SUFFIX}",
        "final": final_seglst_path(wav_path),
        "qwen3": qwen3_seglst_path(wav_path),
        "boundary_final": boundary_final_seglst_path(wav_path),
    }


def write_seglst(path: Path, segments: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def maybe_write_seglst(
    path: Path,
    segments: list[dict],
    *,
    enabled: bool,
    overwrite: bool,
) -> None:
    if not enabled:
        return
    if path.exists() and not overwrite:
        log.info("Skipping intermediate (exists): %s", path.name)
        return
    write_seglst(path, segments)
    log.info("Wrote intermediate: %s (%d segments)", path.name, len(segments))


def plan_pipeline_stages(
    paths: dict[str, Path],
    *,
    overwrite: bool,
    run_asr: bool,
    run_finalization: bool,
) -> tuple[bool, bool, bool, bool]:
    """Return (skip_all, do_segmentation, do_asr, do_finalize)."""
    final_path = paths["final"]
    qwen3_path = paths["qwen3"]
    boundary_final_path = paths["boundary_final"]

    if not overwrite:
        if run_finalization and boundary_final_path.exists():
            return True, False, False, False
        if run_asr and not run_finalization and qwen3_path.exists():
            return True, False, False, False
        if not run_asr and final_path.exists():
            return True, False, False, False

    if overwrite:
        return False, True, run_asr, run_finalization

    do_finalize = run_finalization
    do_asr = run_asr and not qwen3_path.exists()
    if do_asr:
        do_segmentation = not final_path.exists()
    elif do_finalize:
        do_segmentation = False
    else:
        do_segmentation = False

    return False, do_segmentation, do_asr, do_finalize


def job_needs_asr(
    wav_path: Path,
    model: str,
    *,
    overwrite: bool,
    run_asr: bool,
    run_finalization: bool,
) -> bool:
    if not run_asr:
        return False
    paths = intermediate_paths(wav_path, model)
    skip_all, _, do_asr, _ = plan_pipeline_stages(
        paths,
        overwrite=overwrite,
        run_asr=run_asr,
        run_finalization=run_finalization,
    )
    return not skip_all and do_asr


def run_pipeline(wav_path: Path, model: str) -> tuple[list[dict], ...]:
    speech = build_seglst(wav_path, model)
    rms_uncovered = combine_with_rms(wav_path, speech)
    split_silence, _, _, _ = split_seglst(
        wav_path,
        rms_uncovered,
        DEFAULT_MAX_SILENCE_S,
        DEFAULT_SPLIT_MARGIN_S,
    )
    merged, _ = apply_touch_merge(split_silence)
    return speech, rms_uncovered, split_silence, merged


def process_wav(
    wav_path: Path,
    model: str,
    *,
    overwrite_final: bool,
    write_intermediates: bool,
    overwrite_intermediates: bool,
    run_asr: bool,
    run_finalization: bool,
    asr,
    asr_batch_size: int,
    language: str | None,
    min_trim_s: float,
    ref_ms: float,
) -> tuple[int, int, int] | None:
    paths = intermediate_paths(wav_path, model)
    final_path = paths["final"]
    qwen3_path = paths["qwen3"]
    boundary_final_path = paths["boundary_final"]

    skip_all, do_segmentation, do_asr, do_finalize = plan_pipeline_stages(
        paths,
        overwrite=overwrite_final,
        run_asr=run_asr,
        run_finalization=run_finalization,
    )
    if skip_all:
        if run_finalization:
            log.info("Skipping (exists): %s", boundary_final_path.name)
        elif run_asr:
            log.info("Skipping (exists): %s", qwen3_path.name)
        else:
            log.info("Skipping (exists): %s", final_path.name)
        return None

    stage_names = [
        name
        for name, enabled in (
            ("segmentation", do_segmentation),
            ("ASR", do_asr),
            ("finalization", do_finalize),
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

        write_seglst(final_path, merged)
        log.info(
            "%s -> %s (%d segments, %.1fs)",
            wav_path.name,
            final_path.name,
            len(merged),
            time.time() - t0,
        )
    elif do_asr:
        if not final_path.is_file():
            log.error(
                "Cannot run ASR for %s: missing %s",
                wav_path.name,
                final_path.name,
            )
            return None
        merged = load_seglst(final_path)
        log.info("Loaded existing %s (%d segments)", final_path.name, len(merged))
    else:
        if not qwen3_path.is_file():
            log.error(
                "Cannot run finalization for %s: missing %s",
                wav_path.name,
                qwen3_path.name,
            )
            return None
        merged = load_seglst(qwen3_path)

    transcribed = 0
    boundary_fixes = 0
    qwen3_segments: list[dict] | None = None

    if do_asr:
        if asr is None:
            raise RuntimeError("ASR model is not loaded")
        session_id = wav_path.parent.name
        qwen_lang = resolve_language(session_id, language)
        to_transcribe = sum(1 for item in merged if should_transcribe(item))
        lang_label = qwen_lang or "auto"
        log.info(
            "ASR %s -> %s (%d model-family segments, lang=%s)",
            final_path.name,
            qwen3_path.name,
            to_transcribe,
            lang_label,
        )
        t_asr = time.time()
        qwen3_segments = transcribe_seglst_segments(
            wav_path,
            merged,
            asr,
            asr_batch_size,
            qwen_lang,
        )
        write_seglst(qwen3_path, qwen3_segments)
        transcribed = to_transcribe
        log.info(
            "Wrote %s (%d segments transcribed, %.1fs)",
            qwen3_path.name,
            transcribed,
            time.time() - t_asr,
        )
    elif do_finalize:
        qwen3_segments = load_seglst(qwen3_path)
        log.info("Loaded existing %s (%d segments)", qwen3_path.name, len(qwen3_segments))

    if do_finalize:
        assert qwen3_segments is not None
        t_fix = time.time()
        trim_params = EdgeTrimParams(ref_ms=ref_ms, min_trim_s=min_trim_s)
        final_segments, onset_fixes, offset_fixes = fix_segment_boundaries(
            wav_path,
            qwen3_segments,
            trim_params=trim_params,
        )
        write_seglst(boundary_final_path, final_segments)
        boundary_fixes = onset_fixes + offset_fixes
        log.info(
            "Finalized %s -> %s (%d onset + %d offset fixes, %.1fs)",
            qwen3_path.name,
            boundary_final_path.name,
            onset_fixes,
            offset_fixes,
            time.time() - t_fix,
        )

    return len(merged), transcribed, boundary_fixes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create {speaker}.seglst.json, {speaker}_qwen3.seglst.json, and "
            "{speaker}_final.seglst.json via the full pipeline (default)."
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
        "--model",
        choices=["model", "silero", "sortformer"],
        default="model",
        help="Speech detection mode (default: model = Silero union Sortformer)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run all pipeline stages from scratch, ignoring existing outputs "
            "({speaker}.seglst.json, {speaker}_qwen3.seglst.json, {speaker}_final.seglst.json)"
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
        "--skip-asr",
        action="store_true",
        help="Only run segmentation; write {speaker}.seglst.json (no Qwen3 or final output)",
    )
    parser.add_argument(
        "--skip-finalization",
        action="store_true",
        help="Run segmentation + ASR but skip boundary fix ({speaker}_final.seglst.json)",
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
    parser.add_argument(
        "--language",
        default=None,
        help="Qwen3 language override (e.g. Arabic, AR); default: inferred from session_id",
    )
    parser.add_argument("--asr-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="ASR slices per inference batch (lower if GPU OOM)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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

    wav_jobs = collect_wav_jobs(input_root, args.session)
    if not wav_jobs:
        log.warning("No WAV files found under %s", input_root)
        return 0

    log.info("Using device: %s", _get_torch_device())
    run_asr = not args.skip_asr
    run_finalization = run_asr and not args.skip_finalization
    asr = None

    t_total = time.time()
    processed = 0
    skipped = 0
    total_segments = 0
    total_transcribed = 0
    total_boundary_fixes = 0

    for wav_path in wav_jobs:
        if asr is None and job_needs_asr(
            wav_path,
            args.model,
            overwrite=args.overwrite,
            run_asr=run_asr,
            run_finalization=run_finalization,
        ):
            log.info("Loading ASR model...")
            asr = build_model(
                args.asr_model,
                args.device,
                args.batch_size,
                args.max_new_tokens,
            )

        result = process_wav(
            wav_path,
            args.model,
            overwrite_final=args.overwrite,
            write_intermediates=args.keep_intermediates,
            overwrite_intermediates=args.overwrite_intermediates,
            run_asr=run_asr,
            run_finalization=run_finalization,
            asr=asr,
            asr_batch_size=args.batch_size,
            language=args.language,
            min_trim_s=args.min_trim,
            ref_ms=args.ref_ms,
        )
        if result is None:
            skipped += 1
        else:
            seg_count, tx_count, fix_count = result
            processed += 1
            total_segments += seg_count
            total_transcribed += tx_count
            total_boundary_fixes += fix_count

    if run_finalization:
        log.info(
            "Done: mode=%s | %d processed, %d skipped, %d segment(s), "
            "%d transcribed, %d boundary fix(es) in %.1fs",
            args.model,
            processed,
            skipped,
            total_segments,
            total_transcribed,
            total_boundary_fixes,
            time.time() - t_total,
        )
    elif run_asr:
        log.info(
            "Done: mode=%s | %d processed, %d skipped, %d segment(s), "
            "%d transcribed in %.1fs",
            args.model,
            processed,
            skipped,
            total_segments,
            total_transcribed,
            time.time() - t_total,
        )
    else:
        log.info(
            "Done: mode=%s | %d processed, %d skipped, %d segment(s) in %.1fs",
            args.model,
            processed,
            skipped,
            total_segments,
            time.time() - t_total,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
