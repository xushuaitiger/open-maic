"""
POST /api/chat — Stateless multi-agent SSE chat endpoint.

Receives full state from client, runs ONE director→agent cycle through
the LangGraph orchestration graph, and streams events as SSE.

Event types streamed:
  thinking    — director/agent loading indicator
  agent_start — agent begins its turn
  text_delta  — spoken text chunk
  action      — whiteboard/slide action
  agent_end   — agent finished its turn
  cue_user    — session hands control back to student
  done        — session finished (includes updated directorState for next call)
  error       — unrecoverable error

Mirrors lib/orchestration/stateless-generate.ts + app/api/chat/route.ts
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode
from app.sse import sse_response
from core.orchestration.stateless_generate import stateless_generate
from core.providers.llm import resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("chat")
router = APIRouter()


class ChatConfigBody(BaseModel):
    agent_ids: list[str] = Field(default_factory=list, alias="agentIds")
    session_type: str | None = Field(default=None, alias="sessionType")

    model_config = {"populate_by_name": True, "extra": "allow"}


class ChatRequestBody(BaseModel):
    """Loose envelope: we still tolerate camelCase coming from the client."""

    messages: list[dict[str, Any]]
    store_state: dict[str, Any] = Field(alias="storeState")
    config: ChatConfigBody
    director_state: dict[str, Any] | None = Field(default=None, alias="directorState")
    model: str | None = None
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")
    thinking: dict[str, Any] | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}


@router.post("/chat")
async def chat(body: ChatRequestBody, request: Request):
    if not body.messages:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "messages is required")
    if not body.config.agent_ids:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "config.agentIds is required")

    if body.base_url:
        err = validate_url_for_ssrf(body.base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    model_str = body.model or get_settings().default_model
    resolved = resolve_model(model_str, body.api_key or "", body.base_url or "", cfg)

    if not resolved.api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for provider '{resolved.provider_id}'",
            status_code=400,
        )

    # Re-shape into the dict that stateless_generate expects (camelCase).
    request_payload = body.model_dump(by_alias=True, exclude_none=True)

    async def producer():
        try:
            async for event in stateless_generate(request_payload, resolved):
                yield event
        except Exception as exc:
            log.error(
                "Chat stream error (rid=%s): %s",
                getattr(request.state, "request_id", "-"),
                exc,
                exc_info=True,
            )
            yield {"type": "error", "data": {"message": str(exc)}}

    return sse_response(request, producer(), heartbeat_interval=15)
