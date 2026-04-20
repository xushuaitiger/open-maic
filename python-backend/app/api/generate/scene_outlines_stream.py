"""
POST /api/generate/scene-outlines-stream — SSE streaming outline generation.

Streams individual SceneOutline objects as they are parsed from the LLM
response, plus retry events when the LLM returns empty or unparsable output.

Events:
  {type: "outline", data: SceneOutline, index: number}
  {type: "retry",   reason: str, attempt: int, maxAttempts: int}
  {type: "done",    outlines: SceneOutline[]}
  {type: "error",   error: str}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode
from app.sse import sse_response
from core.generation.prompt_strings import OUTLINE_SYSTEM, outline_user_prompt
from core.providers.llm import resolve_model, stream_llm
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("outlines_stream")
router = APIRouter()

_MAX_ATTEMPTS = 2


class _Requirements(BaseModel):
    requirement: str | None = None
    language: str = "zh-CN"

    model_config = {"extra": "allow"}


class OutlineStreamBody(BaseModel):
    requirements: _Requirements | None = None
    requirement: str | None = None
    language: str | None = None
    pdf_text: str | None = Field(default=None, alias="pdfText")
    agents: list[dict[str, Any]] = Field(default_factory=list)
    research_context: str | None = Field(default=None, alias="researchContext")

    model_config = {"populate_by_name": True, "extra": "allow"}


def _extract_new_outlines(buffer: str, already_parsed: int) -> list[dict]:
    """Incrementally extract complete JSON objects from a partial JSON array."""
    results: list[dict] = []
    stripped = re.sub(r"^[\s\S]*?(?=\[)", "", buffer)
    array_start = stripped.find("[")
    if array_start == -1:
        return results

    depth = 0
    object_start = -1
    in_string = False
    escaped = False
    object_count = 0
    i = array_start + 1

    while i < len(stripped):
        ch = stripped[i]
        if escaped:
            escaped = False
        elif ch == "\\" and in_string:
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                if depth == 0:
                    object_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and object_start != -1:
                    object_count += 1
                    if object_count > already_parsed:
                        try:
                            obj = json.loads(stripped[object_start:i + 1])
                            results.append(obj)
                        except json.JSONDecodeError:
                            pass
                    object_start = -1
        i += 1

    return results


@router.post("/generate/scene-outlines-stream")
async def scene_outlines_stream(body: OutlineStreamBody, request: Request):
    requirement = (
        (body.requirements.requirement if body.requirements else None)
        or body.requirement
        or ""
    ).strip()
    language = (
        (body.requirements.language if body.requirements else None)
        or body.language
        or "zh-CN"
    )

    if not requirement:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "requirement is required")

    model_str = request.headers.get("x-model") or get_settings().default_model
    api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved = resolve_model(model_str, api_key, client_base_url, cfg)
    if not resolved.api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for provider '{resolved.provider_id}'",
        )

    teacher = next(
        (a for a in body.agents if a.get("role") == "teacher"),
        body.agents[0] if body.agents else {},
    )
    teacher_ctx = (
        f"Teacher: {teacher.get('name', '')} — {teacher.get('persona', '')}"
        if teacher else ""
    )

    user_prompt = outline_user_prompt(
        requirement=requirement,
        language=language,
        teacher_context=teacher_ctx,
        research_context=body.research_context or "",
        pdf_text=body.pdf_text or "",
    )

    async def producer():
        all_outlines: list[dict] = []
        rid = getattr(request.state, "request_id", "-")
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            buffer = ""
            parsed_count = 0
            this_attempt: list[dict] = []
            try:
                async for chunk in stream_llm(resolved, [
                    {"role": "system", "content": OUTLINE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ]):
                    if await request.is_disconnected():
                        log.info("outlines stream rid=%s: client disconnected", rid)
                        return
                    buffer += chunk
                    new_outlines = _extract_new_outlines(buffer, parsed_count)
                    for outline in new_outlines:
                        if not outline.get("id"):
                            from nanoid import generate as nanoid
                            outline["id"] = f"scene_{nanoid(8)}"
                        this_attempt.append(outline)
                        all_outlines.append(outline)
                        yield {
                            "type": "outline",
                            "data": outline,
                            "index": len(all_outlines) - 1,
                        }
                        parsed_count += 1
            except Exception as exc:
                log.error("Outline stream rid=%s attempt %d error: %s", rid, attempt, exc)
                if attempt < _MAX_ATTEMPTS:
                    yield {
                        "type": "retry",
                        "reason": f"upstream_error: {exc}",
                        "attempt": attempt,
                        "maxAttempts": _MAX_ATTEMPTS,
                    }
                    continue
                yield {"type": "error", "error": str(exc)}
                return

            if this_attempt:
                yield {"type": "done", "outlines": all_outlines}
                return

            # Empty / unparsable response → retry once.
            if attempt < _MAX_ATTEMPTS:
                yield {
                    "type": "retry",
                    "reason": "empty_or_unparsable_response",
                    "attempt": attempt,
                    "maxAttempts": _MAX_ATTEMPTS,
                }
                continue

            yield {
                "type": "error",
                "error": "LLM returned no parsable outlines after retries",
            }
            return

    return sse_response(request, producer(), heartbeat_interval=15)
