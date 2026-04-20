"""Local-disk implementation of `KeyValueStore`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles

from app.config import get_settings


class LocalKeyValueStore:
    """Files at ``{data_root}/.kv/{namespace}/{key}.json``."""

    def _path(self, namespace: str, key: str) -> Path:
        root = get_settings().classrooms_data_dir / ".kv" / namespace
        return root / f"{key}.json"

    async def get(self, namespace: str, key: str) -> dict | None:
        path = self._path(namespace, key)
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            return json.loads(content)
        except FileNotFoundError:
            return None

    async def put(self, namespace: str, key: str, value: dict) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(value, ensure_ascii=False, indent=2))
        os.replace(tmp, path)

    async def delete(self, namespace: str, key: str) -> None:
        path = self._path(namespace, key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
