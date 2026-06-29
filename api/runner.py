from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from api.job_files import APP_ROOT, job_root

log = logging.getLogger(__name__)


def start_job_subprocess(job_id: str) -> None:
    """Launch pipeline in a separate process so NeMo/GPU work cannot block the API."""
    root = job_root(job_id)
    log_path = root / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    subprocess.Popen(
        [sys.executable, "-m", "api.worker", job_id],
        cwd=str(APP_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=False,
    )
    log.info("Started worker subprocess for job %s (log: %s)", job_id, log_path.name)
