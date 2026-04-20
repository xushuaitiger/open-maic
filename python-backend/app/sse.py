"""
SSE (Server-Sent Events) helpers.

Centralises:
  • event encoding (`pack_event`)
  • heartbeat keep-alives (prevents proxies / browsers from killing idle streams)
  • client-disconnect detection (so we don't keep burning LLM tokens after the
    user navigated away)

Usage::

    async def my_producer():
        yield {"type": "thinking", "data": {"message": "..."}}
        async for chunk in stream_llm(...):
            yield {"type": "text_delta", "data": {"content": chunk}}
        yield {"type": "done", "data": {}}

    return sse_response(request, my_producer(), heartbeat_interval=15)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, AsyncIterator, Mapping

from fastapi import Request
from fastapi.responses import StreamingResponse

log = logging.getLogger("sse")

# Comment lines (`: ...\n\n`) are valid SSE keep-alives — the spec says lines
# that start with a colon must be ignored by the client.
_HEARTBEAT_LINE = ": ping\n\n"

DEFAULT_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
}


def pack_event(event: Mapping[str, object] | str, *, event_type: str | None = None) -> str:
    """Encode an event for the SSE wire format.

    `event` can be a dict (serialised as JSON) or already-encoded string.
    """
    if isinstance(event, str):
        data = event
    else:
        data = json.dumps(event, ensure_ascii=False)
    parts = []
    if event_type:
        parts.append(f"event: {event_type}")
    parts.append(f"data: {data}")
    parts.append("")  # final newline
    return "\n".join(parts) + "\n"


async def with_keepalive(
    source: AsyncIterator[dict | str],
    *,
    request: Request | None = None,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """Wrap an async source, interleaving heartbeats and bailing on disconnect.

    - Yields encoded SSE strings.
    - Sends `: ping` every ``heartbeat_interval`` seconds when source is idle.
    - Stops iteration as soon as the underlying request is disconnected.
    """
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=64)
    DONE = object()

    async def pump():
        try:
            async for item in source:
                await queue.put(item)
        except Exception as exc:  # noqa: BLE001
            log.exception("SSE source error: %s", exc)
            await queue.put({"type": "error", "data": {"message": str(exc)}})
        finally:
            await queue.put(DONE)  # type: ignore[arg-type]

    pump_task = asyncio.create_task(pump())

    try:
        while True:
            if request is not None:
                try:
                    if await request.is_disconnected():
                        log.info("SSE client disconnected, aborting stream")
                        break
                except Exception:
                    pass

            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                yield _HEARTBEAT_LINE
                continue

            if item is DONE:
                break
            yield pack_event(item)  # type: ignore[arg-type]
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):
            pass


def sse_response(
    request: Request,
    source: AsyncIterator[dict | str],
    *,
    heartbeat_interval: float = 15.0,
    extra_headers: Mapping[str, str] | None = None,
) -> StreamingResponse:
    """Convenience wrapper returning a fully-configured StreamingResponse."""
    headers = dict(DEFAULT_HEADERS)
    rid = getattr(request.state, "request_id", "")
    if rid:
        headers["X-Request-Id"] = rid
    if extra_headers:
        headers.update(extra_headers)

    return StreamingResponse(
        with_keepalive(source, request=request, heartbeat_interval=heartbeat_interval),
        media_type="text/event-stream",
        headers=headers,
    )
