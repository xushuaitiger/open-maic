"""
MySQL 连接池工具 — 仅保留连接配置，供后续功能扩展使用。

当前状态
────────
用户设置（API keys、provider 选择等）仍然存储在浏览器 localStorage 中，
由前端 Zustand persist 负责管理。MySQL 持久化留待后续迭代实现。

使用方式（待启用时）
──────────────────

    from core.storage.mysql import get_pool, close_pool

    # 获取连接池（懒初始化，首次调用时建立连接）
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()

    # 应用关闭时释放连接池
    await close_pool()

配置
────
通过环境变量或 .env 文件配置（均有默认值，开发环境无需特别设置）：

    MYSQL_HOST=127.0.0.1
    MYSQL_PORT=3306
    MYSQL_USER=root
    MYSQL_PASSWORD=
    MYSQL_DATABASE=openMAIC
    MYSQL_POOL_MIN=1
    MYSQL_POOL_MAX=10
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("mysql")

_pool: Any = None  # aiomysql.Pool — populated lazily by get_pool()


async def get_pool():
    """Return (or lazily create) the module-level aiomysql connection pool.

    Raises ``ImportError`` if ``aiomysql`` is not installed.
    Raises ``pymysql.err.OperationalError`` if the database is unreachable.
    """
    global _pool

    if _pool is not None and not _pool.closed:
        return _pool

    import asyncio
    import aiomysql  # noqa: PLC0415
    from app.config import get_settings  # noqa: PLC0415

    cfg = get_settings()

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
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool (call from app shutdown event)."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.close()
        await _pool.wait_closed()
        log.info("MySQL pool closed")
    _pool = None
