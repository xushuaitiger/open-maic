"""Local-disk implementation of `DocumentStore`.

Mirrors the existing ``classroom_store.persist_classroom`` layout so old data
keeps working::

    {data_root}/{doc_id}/classroom.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles

from app.config import get_settings


class LocalDocumentStore:
    def _path(self, doc_id: str) -> Path:
        return get_settings().classrooms_data_dir / doc_id / "classroom.json"

    async def get(self, doc_id: str) -> dict | None:
        path = self._path(doc_id)
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            return json.loads(content)
        except FileNotFoundError:
            return None

    async def put(self, doc_id: str, value: dict) -> None:
        path = self._path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(value, ensure_ascii=False, indent=2))
        os.replace(tmp, path)

    async def exists(self, doc_id: str) -> bool:
        return self._path(doc_id).exists()
