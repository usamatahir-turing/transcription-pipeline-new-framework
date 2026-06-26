"""Minimal FastAPI wrapper around the transcription pipeline."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.job_files import create_job_files, list_all_jobs, read_result, read_status
from api.runner import start_job_subprocess
from api.store import JobStatus

log = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = APP_ROOT / "api_data"


def _normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "string":
        return None
    return cleaned

app = FastAPI(
    title="Transcription Pipeline API",
    description="Upload a WAV file and receive final segment JSON.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs")
async def list_jobs() -> dict:
    """All jobs (persisted under api_data/; survives API restarts)."""
    jobs = list_all_jobs()
    counts = {status.value: 0 for status in JobStatus}
    for job in jobs:
        status = str(job.get("status", ""))
        if status in counts:
            counts[status] += 1
    return {"counts": counts, "jobs": jobs}


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    speaker: str = Form(...),
    model: str = Form("model"),
    language: str | None = Form(None),
    skip_asr: bool = Form(False),
    skip_finalization: bool = Form(False),
) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload must be a .wav file")

    safe_speaker = Path(speaker).name
    if not safe_speaker:
        raise HTTPException(status_code=400, detail="speaker is required")

    job_id = str(uuid.uuid4())
    create_job_files(
        job_id,
        session_id=session_id,
        speaker=safe_speaker,
        model=model,
        language=_normalize_language(language),
        skip_asr=skip_asr,
        skip_finalization=skip_finalization,
    )

    job_dir = DATA_ROOT / job_id / session_id
    job_dir.mkdir(parents=True, exist_ok=True)
    wav_path = job_dir / f"{safe_speaker}.wav"
    content = await file.read()
    wav_path.write_bytes(content)

    start_job_subprocess(job_id)

    return {"job_id": job_id, "status": JobStatus.QUEUED.value}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = read_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> JSONResponse:
    job = read_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    if status == JobStatus.FAILED.value:
        raise HTTPException(status_code=500, detail=job.get("error") or "Job failed")
    if status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"Job is {status}")

    result = read_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Result file missing")
    return JSONResponse(content=result)
