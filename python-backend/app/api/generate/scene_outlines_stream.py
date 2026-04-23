"""
POST /api/generate/scene-outlines-stream — SSE streaming outline generation.

Mirrors the TypeScript version (app/api/generate/scene-outlines-stream/route.ts)
exactly, including:
  • Template-based prompts via core.generation.prompts.loader
  • Wrapper JSON format: {"languageDirective":"...","outlines":[...]}
    as well as flat array fallback
  • languageDirective SSE event emitted as soon as it is detected
  • order assignment on each outline object
  • interactiveMode → different prompt template
  • Media generation policy from x-image/video-generation-enabled headers
  • User profile (userNickname / userBio) injected into prompt
  • Retry on empty / upstream error (MAX_ATTEMPTS = 3)
  • Heartbeat every 15 s via sse_response

SSE events:
  {type: "languageDirective", data: str}
  {type: "outline",  data: SceneOutline, index: int}
  {type: "retry",    reason: str, attempt: int, maxAttempts: int}
  {type: "done",     outlines: SceneOutline[], languageDirective: str}
  {type: "error",    error: str}
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
from core.generation.prompts.loader import build_prompt
from core.providers.key_resolver import resolve_llm_key, resolve_llm_url
from core.providers.llm import resolve_model, stream_llm
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("outlines_stream")
router = APIRouter()

_MAX_ATTEMPTS = 3
_PROMPT_ID_STANDARD = "requirements-to-outlines"
_PROMPT_ID_INTERACTIVE = "interactive-outlines"   # falls back to standard if missing
_DEFAULT_LANGUAGE_DIRECTIVE = "Teach in the language that matches the user requirement."


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class _Requirements(BaseModel):
    requirement: str | None = None
    language: str = "zh-CN"
    interactive_mode: bool | None = Field(default=None, alias="interactiveMode")
    user_nickname: str | None = Field(default=None, alias="userNickname")
    user_bio: str | None = Field(default=None, alias="userBio")

    model_config = {"extra": "allow", "populate_by_name": True}


class OutlineStreamBody(BaseModel):
    # Nested requirements object (primary frontend format)
    requirements: _Requirements | None = None
    # Flat fields accepted for backward compat / direct API calls
    requirement: str | None = None
    language: str | None = None
    pdf_text: str | None = Field(default=None, alias="pdfText")
    agents: list[dict[str, Any]] = Field(default_factory=list)
    research_context: str | None = Field(default=None, alias="researchContext")

    model_config = {"populate_by_name": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# JSON parsing helpers  (matches TS extractNewOutlines + extractLanguageDirective)
# ---------------------------------------------------------------------------

def _extract_language_directive(buffer: str) -> str | None:
    """
    Extract languageDirective from the wrapper format:
      {"languageDirective":"...","outlines":[...]}
    Mirrors TS extractLanguageDirective().
    """
    m = re.search(r'"languageDirective"\s*:\s*"((?:[^"\\]|\\.)*)"', buffer)
    if not m:
        return None
    try:
        return json.loads(f'"{m.group(1)}"')
    except Exception:
        return m.group(1)


def _extract_new_outlines(buffer: str, already_parsed: int) -> list[dict]:
    """
    Incrementally extract complete JSON objects from a partially-streamed
    JSON response.

    Supports:
      • Flat array:   [{...}, {...}]
      • Wrapper obj:  {"languageDirective":"...","outlines":[{...}]}

    Mirrors TS extractNewOutlines().
    """
    results: list[dict] = []

    # Strip markdown fencing — keep from first [ or {
    stripped = re.sub(r"^[\s\S]*?(?=[\[{])", "", buffer)

    # Locate the outlines array
    array_start = -1
    outlines_key_idx = stripped.find('"outlines"')
    if outlines_key_idx >= 0:
        # Wrapper format: find [ after "outlines":
        array_start = stripped.find("[", outlines_key_idx)
    else:
        # Flat array fallback
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


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_outline_prompt(
    *,
    requirement: str,
    language: str,
    pdf_text: str,
    research_context: str,
    teacher_context: str,
    user_profile: str,
    media_generation_policy: str,
    interactive_mode: bool,
) -> dict[str, str] | None:
    """
    Build system + user prompt using file-based templates.
    Falls back from interactive-outlines to requirements-to-outlines if the
    interactive template is not present.
    """
    prompt_id = _PROMPT_ID_INTERACTIVE if interactive_mode else _PROMPT_ID_STANDARD
    variables = {
        "requirement": requirement,
        "language": language,
        "pdfContent": pdf_text[:12000] if pdf_text else "None",
        "availableImages": "No images available",   # pdfImages vision not yet wired
        "researchContext": research_context or "None",
        "teacherContext": teacher_context,
        "userProfile": user_profile,
        "mediaGenerationPolicy": media_generation_policy,
    }

    prompts = build_prompt(prompt_id, variables)
    if prompts is None and interactive_mode:
        # Fallback: interactive-outlines template not present
        log.warning(
            "Template '%s' not found; falling back to '%s'",
            prompt_id, _PROMPT_ID_STANDARD,
        )
        prompts = build_prompt(_PROMPT_ID_STANDARD, variables)

    return prompts


# ---------------------------------------------------------------------------
# Main route
# ---------------------------------------------------------------------------

@router.post("/generate/scene-outlines-stream")
async def scene_outlines_stream(body: OutlineStreamBody, request: Request):
    # ── Extract fields from nested requirements or flat body ─────────────────
    req_obj = body.requirements
    requirement = (
        (req_obj.requirement if req_obj else None) or body.requirement or ""
    ).strip()
    language = (
        (req_obj.language if req_obj else None) or body.language or "zh-CN"
    )
    interactive_mode = bool(req_obj.interactive_mode if req_obj else False)
    user_nickname = req_obj.user_nickname if req_obj else None
    user_bio = req_obj.user_bio if req_obj else None

    if not requirement:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "requirement is required")

    # ── Model resolution ────────────────────────────────────────────────────
    model_str = request.headers.get("x-model") or get_settings().default_model
    client_key = request.headers.get("x-api-key", "")
    client_url = request.headers.get("x-base-url", "")

    if client_url:
        err = validate_url_for_ssrf(client_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()

    # Extract provider_id for key resolver
    parts = model_str.split(":", 1) if model_str else []
    provider_id = parts[0] if parts else "openai"
    api_key = await resolve_llm_key(provider_id, client_key, cfg)
    base_url = await resolve_llm_url(provider_id, client_url, cfg)

    resolved = resolve_model(model_str, api_key, base_url, cfg)
    if not resolved.api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for provider '{resolved.provider_id}'",
        )

    # ── Media generation policy (from headers) ──────────────────────────────
    image_enabled = request.headers.get("x-image-generation-enabled") == "true"
    video_enabled = request.headers.get("x-video-generation-enabled") == "true"
    if not image_enabled and not video_enabled:
        media_policy = (
            "**IMPORTANT: Do NOT include any mediaGenerations in the outlines. "
            "Both image and video generation are disabled.**"
        )
    elif not image_enabled:
        media_policy = (
            "**IMPORTANT: Do NOT include any image mediaGenerations (type: \"image\") "
            "in the outlines. Image generation is disabled. Video generation is allowed.**"
        )
    elif not video_enabled:
        media_policy = (
            "**IMPORTANT: Do NOT include any video mediaGenerations (type: \"video\") "
            "in the outlines. Video generation is disabled. Image generation is allowed.**"
        )
    else:
        media_policy = ""

    # ── User profile ────────────────────────────────────────────────────────
    if user_nickname or user_bio:
        user_profile = (
            "## Student Profile\n\n"
            f"Student: {user_nickname or 'Unknown'}"
            + (f" — {user_bio}" if user_bio else "")
            + "\n\nConsider this student's background when designing the course. "
            "Adapt difficulty, examples, and teaching approach accordingly.\n\n---"
        )
    else:
        user_profile = ""

    # ── Teacher context ─────────────────────────────────────────────────────
    teacher = next(
        (a for a in body.agents if a.get("role") == "teacher"),
        body.agents[0] if body.agents else None,
    )
    if teacher:
        t_name = teacher.get("name", "")
        t_persona = teacher.get("persona", "")
        teacher_context = f"## Teaching Team\n\nTeacher: {t_name} — {t_persona}"
    else:
        teacher_context = ""

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompts = _build_outline_prompt(
        requirement=requirement,
        language=language,
        pdf_text=body.pdf_text or "",
        research_context=body.research_context or "",
        teacher_context=teacher_context,
        user_profile=user_profile,
        media_generation_policy=media_policy,
        interactive_mode=interactive_mode,
    )
    if not prompts:
        raise ApiException(ErrorCode.INTERNAL_ERROR, "Prompt template not found")

    log.info(
        'Generating outlines: "%s" [model=%s interactive=%s]',
        requirement[:60],
        model_str,
        interactive_mode,
    )

    # ── SSE producer ─────────────────────────────────────────────────────────
    async def producer():
        rid = getattr(request.state, "request_id", "-")
        all_outlines: list[dict] = []
        language_directive: str | None = None
        last_error: str = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            buffer = ""
            parsed_count = 0
            this_attempt: list[dict] = []

            try:
                messages = [
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user",   "content": prompts["user"]},
                ]
                async for chunk in stream_llm(resolved, messages):
                    if await request.is_disconnected():
                        log.info("outlines stream rid=%s: client disconnected", rid)
                        return

                    buffer += chunk

                    # ── languageDirective: extract and emit once ──────────
                    if language_directive is None:
                        language_directive = _extract_language_directive(buffer)
                        if language_directive:
                            yield {
                                "type": "languageDirective",
                                "data": language_directive,
                            }

                    # ── Incremental outline parsing ───────────────────────
                    new_outlines = _extract_new_outlines(buffer, parsed_count)
                    for outline in new_outlines:
                        from nanoid import generate as _nanoid
                        if not outline.get("id"):
                            outline["id"] = f"scene_{_nanoid(8)}"
                        # Always assign order from accumulated position
                        outline["order"] = len(all_outlines) + 1
                        all_outlines.append(outline)
                        this_attempt.append(outline)
                        yield {
                            "type": "outline",
                            "data": outline,
                            "index": len(all_outlines) - 1,
                        }
                        parsed_count += 1

            except Exception as exc:
                last_error = str(exc)
                log.error(
                    "Outline stream rid=%s attempt %d/%d error: %s",
                    rid, attempt, _MAX_ATTEMPTS, exc,
                )
                if attempt < _MAX_ATTEMPTS:
                    yield {
                        "type": "retry",
                        "reason": f"upstream_error: {exc}",
                        "attempt": attempt,
                        "maxAttempts": _MAX_ATTEMPTS,
                    }
                    # Reset accumulators so retry starts fresh
                    all_outlines.clear()
                    language_directive = None
                    continue
                yield {"type": "error", "error": last_error}
                return

            if this_attempt:
                # Success — emit done with all outlines + languageDirective
                yield {
                    "type": "done",
                    "outlines": all_outlines,
                    "languageDirective": (
                        language_directive or _DEFAULT_LANGUAGE_DIRECTIVE
                    ),
                }
                return

            # Empty / unparsable response → retry
            last_error = "LLM returned no parsable outlines"
            if attempt < _MAX_ATTEMPTS:
                log.warning(
                    "Empty outlines rid=%s attempt %d/%d, retrying...",
                    rid, attempt, _MAX_ATTEMPTS,
                )
                yield {
                    "type": "retry",
                    "reason": "empty_or_unparsable_response",
                    "attempt": attempt,
                    "maxAttempts": _MAX_ATTEMPTS,
                }
                all_outlines.clear()
                language_directive = None
                continue

        log.error("Outline generation failed after %d attempts: %s", _MAX_ATTEMPTS, last_error)
        yield {
            "type": "error",
            "error": f"Failed to generate outlines after {_MAX_ATTEMPTS} attempts: {last_error}",
        }

    return sse_response(request, producer(), heartbeat_interval=15)
