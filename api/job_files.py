"""File-based job state under api_data/{job_id}/."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.store import JobStatus

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = APP_ROOT / "api_data"


def job_root(job_id: str) -> Path:
    return DATA_ROOT / job_id


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def create_job_files(
    job_id: str,
    *,
    session_id: str,
    speaker: str,
    model: str,
    language: str | None,
    skip_asr: bool,
    skip_finalization: bool,
) -> Path:
    root = job_root(job_id)
    root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        root / "meta.json",
        {
            "job_id": job_id,
            "session_id": session_id,
            "speaker": speaker,
            "model": model,
            "language": language,
            "skip_asr": skip_asr,
            "skip_finalization": skip_finalization,
            "created_at": created_at,
        },
    )
    _write_json(
        root / "status.json",
        {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "created_at": created_at,
            "session_id": session_id,
            "speaker": speaker,
            "segment_count": 0,
            "transcribed_count": 0,
            "boundary_fixes": 0,
            "error": None,
        },
    )
    return root


def read_status(job_id: str) -> dict[str, Any] | None:
    return _read_json(job_root(job_id) / "status.json")


def read_meta(job_id: str) -> dict[str, Any] | None:
    return _read_json(job_root(job_id) / "meta.json")


def read_result(job_id: str) -> list[dict[str, Any]] | None:
    data = _read_json(job_root(job_id) / "result.json")
    if data is None:
        return None
    if isinstance(data, list):
        return data
    return None


def list_all_jobs() -> list[dict[str, Any]]:
    if not DATA_ROOT.is_dir():
        return []
    jobs: list[dict[str, Any]] = []
    for path in DATA_ROOT.iterdir():
        if not path.is_dir():
            continue
        status = _read_json(path / "status.json")
        if status is not None:
            jobs.append(status)
    jobs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return jobs
