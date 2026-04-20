"""
Settings persistence — MySQL 5.6 + Fernet encryption.

Why MySQL instead of SQLite:
  • MySQL 5.6 is the target deployment environment.
  • Supports concurrent read/write from multiple uvicorn workers.
  • Familiar ops tooling (backups, monitoring, replication).

Why NOT a full ORM (SQLAlchemy/Tortoise):
  • The schema is tiny — 2 tables, 4 columns each.
  • aiomysql is a pure-Python async driver with zero native deps.
  • Keeping it thin avoids ORM version conflicts with FastAPI/Pydantic.

MySQL 5.6 compatibility notes used throughout this file:
  • No JSON column type      → use LONGTEXT
  • No ON CONFLICT clause    → use INSERT ... ON DUPLICATE KEY UPDATE
  • Use %s placeholders      → NOT ? (that's SQLite/psycopg2)
  • ENGINE=InnoDB + utf8mb4  → required for emoji / multi-byte text
  • DOUBLE for timestamps    → MySQL has no REAL type affinity like SQLite

Encryption key resolution order
  1. settings_encryption_key in .env / env var
  2. SETTINGS_ENCRYPTION_KEY env var (legacy / backward-compat)
  3. Auto-generated key persisted to  <data_dir>/.settings_key  (chmod 600).
     On first boot the key is created; subsequent boots load the same file.
     WARNING: losing this file makes all stored API keys unrecoverable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("settings_db")

# ---------------------------------------------------------------------------
# Data-dir + key-file helpers (unchanged from SQLite version)
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    from app.config import get_settings  # lazy to avoid circular
    return Path(get_settings().data_dir)


def _key_file_path() -> Path:
    return _data_root() / ".settings_key"


# ---------------------------------------------------------------------------
# Fernet encryption helpers
# ---------------------------------------------------------------------------

_cached_fernet: Any = None


def _get_fernet() -> Any:
    global _cached_fernet
    if _cached_fernet is not None:
        return _cached_fernet

    from cryptography.fernet import Fernet  # noqa: PLC0415

    # 1. pydantic-settings field
    from app.config import get_settings
    raw_key = get_settings().settings_encryption_key.strip()

    # 2. legacy bare env var
    if not raw_key:
        raw_key = os.environ.get("SETTINGS_ENCRYPTION_KEY", "").strip()

    if raw_key:
        _cached_fernet = Fernet(raw_key.encode())
        return _cached_fernet

    # 3. auto-generated key file
    key_file = _key_file_path()
    if key_file.exists():
        raw_key = key_file.read_text().strip()
        _cached_fernet = Fernet(raw_key.encode())
        return _cached_fernet

    log.warning(
        "No SETTINGS_ENCRYPTION_KEY configured. "
        "Auto-generating a Fernet key and saving to %s. "
        "Back this file up — losing it makes stored API keys unrecoverable.",
        key_file,
    )
    key_file.parent.mkdir(parents=True, exist_ok=True)
    new_key = Fernet.generate_key()
    key_file.write_bytes(new_key)
    try:
        key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass  # Windows
    _cached_fernet = Fernet(new_key)
    return _cached_fernet


def _encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def _decrypt(token: str) -> str:
    return _get_fernet().decrypt(token.encode()).decode()


# ---------------------------------------------------------------------------
# MySQL connection pool
# ---------------------------------------------------------------------------

_pool: Any = None           # aiomysql.Pool
_pool_lock = asyncio.Lock()


async def _get_pool():
    """Return (or lazily create) the module-level aiomysql connection pool."""
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool

    import aiomysql  # noqa: PLC0415
    from app.config import get_settings

    cfg = get_settings()
    async with _pool_lock:
        # Double-checked locking: another coroutine may have created it.
        if _pool is not None and not _pool.closed:
            return _pool

        log.info(
            "Connecting to MySQL %s:%s/%s (pool %s–%s)",
            cfg.mysql_host, cfg.mysql_port, cfg.mysql_database,
            cfg.mysql_pool_min, cfg.mysql_pool_max,
        )
        _pool = await aiomysql.create_pool(
            host=cfg.mysql_host,
            port=cfg.mysql_port,
            user=cfg.mysql_user,
            password=cfg.mysql_password,
            db=cfg.mysql_database,
            charset="utf8mb4",
            autocommit=False,
            minsize=cfg.mysql_pool_min,
            maxsize=cfg.mysql_pool_max,
            connect_timeout=10,
        )
        await _ensure_schema(_pool)
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool (call from app shutdown)."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.close()
        await _pool.wait_closed()
        log.info("MySQL pool closed")
    _pool = None


# ---------------------------------------------------------------------------
# Schema bootstrap  (MySQL 5.6 compatible DDL)
# ---------------------------------------------------------------------------

_DDL_PROFILES = """
CREATE TABLE IF NOT EXISTS settings_profiles (
    profile     VARCHAR(128)  NOT NULL,
    created_at  DOUBLE        NOT NULL,
    updated_at  DOUBLE        NOT NULL,
    PRIMARY KEY (profile)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC
"""

_DDL_DATA = """
CREATE TABLE IF NOT EXISTS settings_data (
    profile     VARCHAR(128)  NOT NULL,
    namespace   VARCHAR(128)  NOT NULL,
    data_enc    LONGTEXT      NOT NULL,
    updated_at  DOUBLE        NOT NULL,
    PRIMARY KEY (profile, namespace)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC
"""


async def _ensure_schema(pool) -> None:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DDL_PROFILES)
            await cur.execute(_DDL_DATA)
        await conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def save_settings(
    data: dict[str, Any],
    *,
    profile: str = "default",
    namespace: str = "all",
) -> None:
    """Encrypt and persist *data* to MySQL."""
    pool = await _get_pool()
    now = time.time()
    enc = _encrypt(json.dumps(data, ensure_ascii=False))

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Upsert profile row — MySQL 5.6: INSERT ... ON DUPLICATE KEY UPDATE
            await cur.execute(
                """
                INSERT INTO settings_profiles (profile, created_at, updated_at)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE updated_at = VALUES(updated_at)
                """,
                (profile, now, now),
            )
            # Upsert data row
            await cur.execute(
                """
                INSERT INTO settings_data (profile, namespace, data_enc, updated_at)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    data_enc   = VALUES(data_enc),
                    updated_at = VALUES(updated_at)
                """,
                (profile, namespace, enc, now),
            )
        await conn.commit()
    log.info("Settings saved (profile=%s namespace=%s)", profile, namespace)


async def load_settings(
    *,
    profile: str = "default",
    namespace: str = "all",
) -> dict[str, Any] | None:
    """Load and decrypt settings. Returns None if not found."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT data_enc FROM settings_data WHERE profile=%s AND namespace=%s",
                (profile, namespace),
            )
            row = await cur.fetchone()

    if not row:
        return None
    try:
        return json.loads(_decrypt(row[0]))
    except Exception as exc:
        log.error(
            "Failed to decrypt settings (profile=%s ns=%s): %s",
            profile, namespace, exc,
        )
        return None


async def delete_settings(
    *,
    profile: str = "default",
    namespace: str | None = None,
) -> int:
    """Delete settings rows. Returns number of deleted rows.

    If *namespace* is None, all namespaces for the profile are removed.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            if namespace is None:
                await cur.execute(
                    "DELETE FROM settings_data WHERE profile=%s", (profile,)
                )
                count = cur.rowcount
                await cur.execute(
                    "DELETE FROM settings_profiles WHERE profile=%s", (profile,)
                )
            else:
                await cur.execute(
                    "DELETE FROM settings_data WHERE profile=%s AND namespace=%s",
                    (profile, namespace),
                )
                count = cur.rowcount
        await conn.commit()
    log.info("Settings deleted (profile=%s ns=%s rows=%d)", profile, namespace, count)
    return count


async def list_profiles() -> list[dict[str, Any]]:
    """Return metadata for all existing profiles, newest first."""
    import aiomysql  # noqa: PLC0415
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT profile, created_at, updated_at "
                "FROM settings_profiles ORDER BY updated_at DESC"
            )
            rows = await cur.fetchall()
    return list(rows)


# ---------------------------------------------------------------------------
# Key-masking helper (presentation layer — lives here so rules are centralised)
# ---------------------------------------------------------------------------

def mask_api_keys(obj: Any, *, keep: int = 4) -> Any:
    """Recursively replace API key values with masked versions like ``••••abcd``."""
    if isinstance(obj, dict):
        return {k: _maybe_mask(k, v, keep=keep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_api_keys(item, keep=keep) for item in obj]
    return obj


def _maybe_mask(key: str, value: Any, *, keep: int) -> Any:
    lower = key.lower()
    if any(p in lower for p in ("apikey", "api_key")):
        if isinstance(value, str) and value:
            visible = value[-keep:] if len(value) >= keep else value
            return "•" * max(8, len(value) - keep) + visible
    return mask_api_keys(value, keep=keep)
