"""
Background job runner for classroom generation.

Uses asyncio.create_task for fire-and-forget execution.
Deduplicates concurrent calls to the same job ID.
"""

from __future__ import annotations

import asyncio
import logging

from core.generation.classroom_generator import generate_classroom
from core.storage.job_store import (
    mark_job_failed,
    mark_job_running,
    mark_job_succeeded,
    update_job_progress,
)

log = logging.getLogger("job_runner")
_running_jobs: dict[str, asyncio.Task] = {}


def run_classroom_job(job_id: str, input_data: dict, base_url: str) -> asyncio.Task:
    """
    Start a classroom generation job as a background asyncio task.
    Returns the existing task if one is already running for this job_id.
    """
    if job_id in _running_jobs:
        return _running_jobs[job_id]

    task = asyncio.create_task(_run(job_id, input_data, base_url))
    _running_jobs[job_id] = task
    task.add_done_callback(lambda _: _running_jobs.pop(job_id, None))
    return task


async def _run(job_id: str, input_data: dict, base_url: str) -> None:
    try:
        await mark_job_running(job_id)

        result = await generate_classroom(
            input_data,
            base_url,
            on_progress=lambda p: update_job_progress(job_id, p),
        )

        await mark_job_succeeded(job_id, result)
    except Exception as exc:
        message = str(exc)
        log.error("Classroom generation job %s failed: %s", job_id, message, exc_info=True)
        try:
            await mark_job_failed(job_id, message)
        except Exception:
            log.exception("Failed to persist failure for job %s", job_id)
