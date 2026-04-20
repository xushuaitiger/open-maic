"""
Classroom generation job store.

Jobs are persisted as JSON files under:
  {CLASSROOMS_DIR}/.jobs/{job_id}.json

Per-job asyncio locks prevent read-modify-write races.
Stale jobs (no update for 30 min) are auto-failed on read.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiofiles

from app.config import get_settings

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

JobStatus = Literal["queued", "running", "succeeded", "failed"]
GenerationStep = Literal[
    "queued",
    "initializing",
    "researching",
    "generating_outlines",
    "generating_scenes",
    "generating_media",
    "generating_tts",
    "persisting",
    "completed",
    "failed",
]

_STALE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Per-job asyncio locks
_job_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _jobs_dir() -> Path:
    return get_settings().classrooms_data_dir / ".jobs"


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def is_valid_job_id(job_id: str) -> bool:
    return bool(_ID_RE.match(job_id)) and len(job_id) <= 64


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------

async def _get_lock(job_id: str) -> asyncio.Lock:
    async with _locks_lock:
        if job_id not in _job_locks:
            _job_locks[job_id] = asyncio.Lock()
        return _job_locks[job_id]


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

async def _write_job(job_id: str, data: dict) -> None:
    path = _job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------

def _mark_stale_if_needed(job: dict) -> dict:
    if job.get("status") != "running":
        return job
    updated_at = job.get("updatedAt", "")
    if not updated_at:
        return job
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        if age > _STALE_TIMEOUT_SECONDS:
            now = datetime.now(timezone.utc).isoformat()
            return {
                **job,
                "status": "failed",
                "step": "failed",
                "message": "Job appears stale (no progress update for 30 minutes)",
                "error": "Stale job: process may have restarted during generation",
                "completedAt": now,
                "updatedAt": now,
            }
    except (ValueError, TypeError):
        pass
    return job


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_job(job_id: str, input_data: dict) -> dict:
    """Create and persist a new queued job."""
    now = _now_iso()
    req_preview = input_data.get("requirement", "")
    if len(req_preview) > 200:
        req_preview = req_preview[:197] + "..."
    pdf_content = input_data.get("pdf_content") or {}

    job = {
        "id": job_id,
        "status": "queued",
        "step": "queued",
        "progress": 0,
        "message": "Classroom generation job queued",
        "createdAt": now,
        "updatedAt": now,
        "inputSummary": {
            "requirementPreview": req_preview,
            "language": input_data.get("language") or "zh-CN",
            "hasPdf": bool(pdf_content),
            "pdfTextLength": len(pdf_content.get("text", "")),
            "pdfImageCount": len(pdf_content.get("images", [])),
        },
        "scenesGenerated": 0,
        "totalScenes": None,
        "result": None,
        "error": None,
    }
    await _write_job(job_id, job)
    return job


async def read_job(job_id: str) -> dict | None:
    try:
        async with aiofiles.open(_job_path(job_id), "r", encoding="utf-8") as f:
            content = await f.read()
        job = json.loads(content)
        return _mark_stale_if_needed(job)
    except FileNotFoundError:
        return None


async def _update_job(job_id: str, patch: dict) -> dict:
    lock = await _get_lock(job_id)
    async with lock:
        existing = await read_job(job_id)
        if not existing:
            raise ValueError(f"Job not found: {job_id}")
        updated = {**existing, **patch, "updatedAt": _now_iso()}
        await _write_job(job_id, updated)
        return updated


async def mark_job_running(job_id: str) -> dict:
    return await _update_job(job_id, {
        "status": "running",
        "startedAt": _now_iso(),
        "message": "Classroom generation started",
    })


async def update_job_progress(job_id: str, progress: dict) -> dict:
    return await _update_job(job_id, {
        "status": "running",
        "step": progress.get("step"),
        "progress": progress.get("progress", 0),
        "message": progress.get("message", ""),
        "scenesGenerated": progress.get("scenes_generated", 0),
        "totalScenes": progress.get("total_scenes"),
    })


async def mark_job_succeeded(job_id: str, result: dict) -> dict:
    return await _update_job(job_id, {
        "status": "succeeded",
        "step": "completed",
        "progress": 100,
        "message": "Classroom generation completed",
        "completedAt": _now_iso(),
        "scenesGenerated": result.get("scenes_count", 0),
        "result": {
            "classroomId": result.get("id"),
            "url": result.get("url"),
            "scenesCount": result.get("scenes_count", 0),
        },
    })


async def mark_job_failed(job_id: str, error: str) -> dict:
    return await _update_job(job_id, {
        "status": "failed",
        "step": "failed",
        "message": "Classroom generation failed",
        "completedAt": _now_iso(),
        "error": error,
    })
