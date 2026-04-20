"""
Provider call resilience: retries with exponential backoff + provider-level
concurrency limiting.

Intentionally light-weight (no tenacity dep — keeps requirements small).

Usage::

    from core.providers.retry import with_retry, provider_semaphore

    async with provider_semaphore("openai"):
        result = await with_retry(
            lambda: client.chat.completions.create(...),
            attempts=3,
            label="openai.chat",
        )
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, TypeVar

import httpx

log = logging.getLogger("provider_retry")

T = TypeVar("T")


# ── Concurrency limiter ─────────────────────────────────────────────────────

# Per-provider semaphores keep us under provider rate caps even when many
# requests are processed in parallel (e.g. scenes).  Defaults are conservative
# and can be raised once the deploy environment allows.
_DEFAULT_CONCURRENCY = 6
_PROVIDER_LIMITS: dict[str, int] = {
    "openai": 6,
    "anthropic": 6,
    "google": 4,
    "deepseek": 8,
    "qwen": 6,
    "kimi": 4,
    "glm": 6,
    "siliconflow": 6,
    "doubao": 6,
    "minimax": 4,
}

_semaphores: dict[str, asyncio.Semaphore] = {}


def provider_semaphore(provider_id: str) -> asyncio.Semaphore:
    sem = _semaphores.get(provider_id)
    if sem is None:
        limit = _PROVIDER_LIMITS.get(provider_id, _DEFAULT_CONCURRENCY)
        sem = asyncio.Semaphore(limit)
        _semaphores[provider_id] = sem
    return sem


# ── Retry helpers ───────────────────────────────────────────────────────────

# These exception types are usually safe to retry: transient network blips,
# 429 rate-limit responses, 5xx upstream failures.
_RETRY_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_HTTP_STATUSES
    msg = str(exc).lower()
    if "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg:
        return True
    if "timeout" in msg or "temporarily unavailable" in msg:
        return True
    # SDK-specific: openai.RateLimitError / anthropic.RateLimitError /
    # google APICore exceptions all expose a `.status_code` attribute.
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _RETRY_HTTP_STATUSES:
        return True
    return False


async def with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.6,
    max_delay: float = 6.0,
    label: str = "call",
) -> T:
    """Run ``func`` with bounded exponential backoff retries.

    Only retries on transient failures (timeouts, 5xx, 429).
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 — we re-raise non-retryable
            last_exc = exc
            if attempt >= attempts or not _is_retryable(exc):
                raise
            sleep_for = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_for *= 0.6 + random.random() * 0.8  # jitter
            log.warning(
                "%s attempt %d/%d failed (%s: %s) — retrying in %.2fs",
                label, attempt, attempts, type(exc).__name__, exc, sleep_for,
            )
            await asyncio.sleep(sleep_for)
    # Unreachable, but keeps type-checkers happy
    assert last_exc is not None
    raise last_exc
