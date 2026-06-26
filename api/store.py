from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: str
    session_id: str
    speaker: str
    error: str | None = None
    result: list[dict[str, Any]] | None = None
    segment_count: int = 0
    transcribed_count: int = 0
    boundary_fixes: int = 0


@dataclass
class JobStore:
    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, session_id: str, speaker: str) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            speaker=speaker,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            return job

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


job_store = JobStore()
