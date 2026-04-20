"""
Classroom persistence layer.

Stores classrooms as JSON files under:
  {CLASSROOMS_DIR}/{classroom_id}/classroom.json

This mirrors the original TypeScript classroom-storage.ts behaviour.
Swap this module for a DB-backed implementation without touching the API layer.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import aiofiles

from app.config import get_settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _classrooms_dir() -> Path:
    return get_settings().classrooms_data_dir


def _classroom_path(classroom_id: str) -> Path:
    return _classrooms_dir() / classroom_id / "classroom.json"


def is_valid_classroom_id(classroom_id: str) -> bool:
    return bool(_ID_RE.match(classroom_id)) and len(classroom_id) <= 64


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

async def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def persist_classroom(classroom_data: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Write classroom to disk and return {id, url}."""
    classroom_id: str = classroom_data["id"]
    url = f"{base_url}/api/classroom?id={classroom_id}"
    enriched = {**classroom_data, "url": url}
    await _write_json_atomic(_classroom_path(classroom_id), enriched)
    return {"id": classroom_id, "url": url}


async def read_classroom(classroom_id: str) -> dict[str, Any] | None:
    """Read classroom from disk. Returns None if not found."""
    path = _classroom_path(classroom_id)
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
        return json.loads(content)
    except FileNotFoundError:
        return None


async def get_classroom_media_path(
    classroom_id: str,
    path_segments: list[str],
) -> Path | None:
    """
    Resolve a media file path within a classroom directory.
    Returns None if the path is invalid or outside the classroom dir.
    """
    if not is_valid_classroom_id(classroom_id):
        return None

    if any(".." in seg or "\x00" in seg for seg in path_segments):
        return None

    sub_dir = path_segments[0] if path_segments else ""
    if sub_dir not in ("media", "audio"):
        return None

    classrooms_dir = _classrooms_dir()
    file_path = classrooms_dir / classroom_id / Path(*path_segments)
    resolved_base = (classrooms_dir / classroom_id).resolve()

    try:
        real_path = file_path.resolve()
        if not str(real_path).startswith(str(resolved_base)):
            return None
        if not real_path.is_file():
            return None
        return real_path
    except (OSError, ValueError):
        return None
