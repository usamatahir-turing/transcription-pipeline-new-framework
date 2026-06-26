"""Run one pipeline job in an isolated process (called by the API server)."""
from __future__ import annotations

import json
import logging
import sys

from api.job_files import job_root, read_meta
from api.store import JobStatus

log = logging.getLogger(__name__)


def _write_status(job_id: str, payload: dict) -> None:
    path = job_root(job_id) / "status.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _write_result(job_id: str, segments: list[dict]) -> None:
    path = job_root(job_id) / "result.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() == "string":
        return None
    return cleaned


def run_worker(job_id: str) -> int:
    from create_segments import (
        boundary_final_seglst_path,
        final_seglst_path,
        load_seglst,
        process_wav,
        qwen3_seglst_path,
    )
    from finalization_scripts.segment_boundary_fix import DEFAULT_MIN_TRIM_S, DEFAULT_REF_MS
    from transcription_scripts.qwen3_asr_gen import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_DEVICE,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_MODEL,
        build_model,
    )

    meta = read_meta(job_id)
    if meta is None:
        log.error("Missing meta.json for job %s", job_id)
        return 1

    status_path = job_root(job_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = JobStatus.RUNNING.value
    _write_status(job_id, status)

    session_id = meta["session_id"]
    speaker = meta["speaker"]
    wav_path = job_root(job_id) / session_id / f"{speaker}.wav"
    if not wav_path.is_file():
        status["status"] = JobStatus.FAILED.value
        status["error"] = f"WAV not found: {wav_path.name}"
        _write_status(job_id, status)
        return 1

    try:
        skip_asr = bool(meta.get("skip_asr", False))
        skip_finalization = bool(meta.get("skip_finalization", False))
        run_asr = not skip_asr
        run_finalization = run_asr and not skip_finalization
        language = _normalize_language(meta.get("language"))
        model = meta.get("model", "model")

        # Segmentation first (Sortformer/Silero), then load ASR — avoids both on GPU at once.
        log.info("Stage 1/2: segmentation")
        seg_result = process_wav(
            wav_path,
            model,
            overwrite_final=True,
            write_intermediates=False,
            overwrite_intermediates=False,
            run_asr=False,
            run_finalization=False,
            asr=None,
            asr_batch_size=DEFAULT_BATCH_SIZE,
            language=language,
            min_trim_s=DEFAULT_MIN_TRIM_S,
            ref_ms=DEFAULT_REF_MS,
        )
        if seg_result is None:
            raise RuntimeError("Segmentation returned no result")

        asr = None
        if run_asr:
            log.info("Stage 2/2: loading ASR model")
            asr = build_model(
                DEFAULT_MODEL,
                DEFAULT_DEVICE,
                DEFAULT_BATCH_SIZE,
                DEFAULT_MAX_NEW_TOKENS,
            )
            result = process_wav(
                wav_path,
                model,
                overwrite_final=False,
                write_intermediates=False,
                overwrite_intermediates=False,
                run_asr=True,
                run_finalization=run_finalization,
                asr=asr,
                asr_batch_size=DEFAULT_BATCH_SIZE,
                language=language,
                min_trim_s=DEFAULT_MIN_TRIM_S,
                ref_ms=DEFAULT_REF_MS,
            )
            if result is None:
                raise RuntimeError("ASR/finalization returned no result")
            seg_count, tx_count, fix_count = result
        else:
            seg_count, tx_count, fix_count = seg_result

        if run_finalization:
            output_path = boundary_final_seglst_path(wav_path)
        elif run_asr:
            output_path = qwen3_seglst_path(wav_path)
        else:
            output_path = final_seglst_path(wav_path)

        segments = load_seglst(output_path)
        _write_result(job_id, segments)

        status["status"] = JobStatus.COMPLETED.value
        status["segment_count"] = seg_count
        status["transcribed_count"] = tx_count
        status["boundary_fixes"] = fix_count
        status["error"] = None
        _write_status(job_id, status)
        log.info("Job %s completed (%d segments)", job_id, len(segments))
        return 0
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        status["status"] = JobStatus.FAILED.value
        status["error"] = str(exc)
        _write_status(job_id, status)
        return 1


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) != 2:
        log.error("Usage: python -m api.worker JOB_ID")
        return 1
    return run_worker(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
