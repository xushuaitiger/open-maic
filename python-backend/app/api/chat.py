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

IMPORTANT: We use list[Any] for messages because the AI SDK v4 UIMessage type
has a `content` field that can be either a string or an array of ContentPart
objects.  Pydantic's stricter list[dict[str, Any]] silently drops messages with
array content, producing an empty list and a 400 error.

We also keep storeState optional to avoid 422 validation errors when the client
omits it in edge cases.
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
from core.providers.key_resolver import resolve_llm_key, resolve_llm_url
from core.providers.llm import resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("chat")
router = APIRouter()


class ChatConfigBody(BaseModel):
    agent_ids: list[str] = Field(default_factory=list, alias="agentIds")
    session_type: str | None = Field(default=None, alias="sessionType")

    model_config = {"populate_by_name": True, "extra": "allow"}


class ChatRequestBody(BaseModel):
    # list[Any] — accepts UIMessage objects where content may be str OR list
    messages: list[Any] = Field(default_factory=list)
    # storeState optional so validation never rejects on missing field
    store_state: Any = Field(default=None, alias="storeState")
    config: ChatConfigBody = Field(default_factory=ChatConfigBody)
    director_state: Any = Field(default=None, alias="directorState")
    model: str | None = None
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")
    thinking: Any = None
    user_profile: Any = Field(default=None, alias="userProfile")

    model_config = {"populate_by_name": True, "extra": "allow"}


@router.post("/chat")
async def chat(body: ChatRequestBody, request: Request):
    if body.messages is None:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "messages is required")
    if not body.config.agent_ids:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "config.agentIds is required")

    cfg = get_server_config()
    model_str = request.headers.get("x-model") or body.model or get_settings().default_model
    client_key = (request.headers.get("x-api-key") or "").strip() or (body.api_key or "")
    client_url = (request.headers.get("x-base-url") or "").strip() or (body.base_url or "")

    if client_url:
        err = validate_url_for_ssrf(client_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    parts = model_str.split(":", 1) if model_str else []
    provider_id = parts[0] if parts else "openai"
    api_key = await resolve_llm_key(provider_id, client_key, cfg)
    base_url = await resolve_llm_url(provider_id, client_url, cfg)
    resolved = resolve_model(model_str, api_key, base_url, cfg)

    if not resolved.api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for provider '{resolved.provider_id}'",
            status_code=400,
        )

    rid = getattr(request.state, "request_id", "-")
    log.info(
        "Chat rid=%s agents=%s turns=%s msgs=%d [model=%s]",
        rid,
        body.config.agent_ids,
        (body.director_state or {}).get("turnCount", 0) if isinstance(body.director_state, dict) else 0,
        len(body.messages),
        model_str,
    )

    # Pass the original body as camelCase dict to stateless_generate
    request_payload = body.model_dump(by_alias=True, exclude_none=True)

    async def producer():
        try:
            async for event in stateless_generate(request_payload, resolved):
                yield event
        except Exception as exc:
            log.error("Chat stream error rid=%s: %s", rid, exc, exc_info=True)
            yield {"type": "error", "data": {"message": str(exc)}}

    return sse_response(request, producer(), heartbeat_interval=15)
