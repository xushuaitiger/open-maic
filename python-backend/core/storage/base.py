"""
Storage backend abstraction.

The original Python rewrite stored everything as JSON files on local disk —
fine for single-process dev, but it doesn't survive horizontal scaling or
container restarts.

This module defines two minimal protocols (`KeyValueStore` for jobs,
`DocumentStore` for classrooms) so we can swap in Redis/S3/Postgres later
without touching any callers.

For backwards compatibility, the existing file-backed implementations
(`job_store.py`, `classroom_store.py`) are still the default — but new code
should depend on the protocols below and resolve a backend through
``get_kv_store()`` / ``get_doc_store()``.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


# ── Protocols ───────────────────────────────────────────────────────────────

@runtime_checkable
class KeyValueStore(Protocol):
    """Async key/value store for short-lived JSON documents (jobs, sessions)."""

    async def get(self, namespace: str, key: str) -> dict | None: ...
    async def put(self, namespace: str, key: str, value: dict) -> None: ...
    async def delete(self, namespace: str, key: str) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    """Async document store for long-lived JSON documents (classrooms)."""

    async def get(self, doc_id: str) -> dict | None: ...
    async def put(self, doc_id: str, value: dict) -> None: ...
    async def exists(self, doc_id: str) -> bool: ...


# ── Backend selection ───────────────────────────────────────────────────────

_kv: KeyValueStore | None = None
_doc: DocumentStore | None = None


def _backend() -> str:
    return os.environ.get("STORAGE_BACKEND", "local").lower().strip() or "local"


def get_kv_store() -> KeyValueStore:
    global _kv
    if _kv is not None:
        return _kv
    backend = _backend()
    if backend == "local":
        from core.storage.local_kv import LocalKeyValueStore
        _kv = LocalKeyValueStore()
    else:
        raise NotImplementedError(
            f"STORAGE_BACKEND={backend!r} is not implemented yet. "
            "Only 'local' is currently supported."
        )
    return _kv


def get_doc_store() -> DocumentStore:
    global _doc
    if _doc is not None:
        return _doc
    backend = _backend()
    if backend == "local":
        from core.storage.local_doc import LocalDocumentStore
        _doc = LocalDocumentStore()
    else:
        raise NotImplementedError(
            f"STORAGE_BACKEND={backend!r} is not implemented yet. "
            "Only 'local' is currently supported."
        )
    return _doc
