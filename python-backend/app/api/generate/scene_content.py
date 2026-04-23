"""
POST /api/generate/scene-content — generate content for a single scene.

Mirrors the TypeScript route (app/api/generate/scene-content/route.ts):
  • Uses file-based prompt templates (slide-content, quiz-content, interactive-html)
  • Reads outline.type (not scene_type)
  • Returns { content, effectiveOutline } flat-spread via api_success
"""

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.generation.prompts.loader import build_prompt
from core.providers.key_resolver import resolve_llm_key, resolve_llm_url
from core.providers.llm import call_llm, resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("scene_content")
router = APIRouter()

_CANVAS_WIDTH = 1000
_CANVAS_HEIGHT = 562   # 16:9 → 1000 × 0.5625


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class SceneContentBody(BaseModel):
    outline: dict[str, Any]
    all_outlines: list[dict[str, Any]] = Field(default_factory=list, alias="allOutlines")
    stage_info: dict[str, Any] = Field(default_factory=dict, alias="stageInfo")
    stage_id: str | None = Field(default=None, alias="stageId")
    agents: list[dict[str, Any]] = Field(default_factory=list)
    language: str | None = None
    language_directive: str | None = Field(default=None, alias="languageDirective")
    pdf_images: list[dict[str, Any]] = Field(default_factory=list, alias="pdfImages")
    image_mapping: dict[str, Any] = Field(default_factory=dict, alias="imageMapping")

    model_config = {"populate_by_name": True, "extra": "ignore"}


# ---------------------------------------------------------------------------
# JSON cleaning helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json(text: str) -> Any:
    return json.loads(_strip_fences(text))


# ---------------------------------------------------------------------------
# Per-type content generators
# ---------------------------------------------------------------------------

async def _gen_slide_content(
    outline: dict[str, Any],
    agents: list[dict[str, Any]],
    language_directive: str,
    resolved,
) -> dict[str, Any]:
    teacher = next((a for a in agents if a.get("role") == "teacher"), None)
    teacher_ctx = ""
    if teacher:
        teacher_ctx = f"\n## Teacher\n{teacher.get('name', '')} — {teacher.get('persona', '')}\n"

    key_points = outline.get("keyPoints", [])
    if isinstance(key_points, list):
        key_points_str = "\n".join(f"  - {p}" for p in key_points)
    else:
        key_points_str = str(key_points)

    prompts = build_prompt("slide-content", {
        "title": outline.get("title", ""),
        "description": outline.get("description", ""),
        "keyPoints": key_points_str,
        "teacherContext": teacher_ctx,
        "assignedImages": "None",
        "canvas_width": _CANVAS_WIDTH,
        "canvas_height": _CANVAS_HEIGHT,
        "languageDirective": language_directive or "",
    })
    if not prompts:
        raise ValueError("slide-content template not found")

    raw = await call_llm(
        resolved,
        [{"role": "system", "content": prompts["system"]},
         {"role": "user",   "content": prompts["user"]}],
    )
    content = _parse_json(raw)
    # Ensure list fields exist
    if "elements" not in content:
        content["elements"] = []
    if "background" not in content:
        content["background"] = {"type": "solid", "color": "#ffffff"}
    return content


async def _gen_quiz_content(
    outline: dict[str, Any],
    resolved,
) -> dict[str, Any]:
    quiz_cfg = outline.get("quizConfig", {})
    key_points = outline.get("keyPoints", [])
    key_points_str = ", ".join(key_points) if isinstance(key_points, list) else str(key_points)

    prompts = build_prompt("quiz-content", {
        "title": outline.get("title", ""),
        "description": outline.get("description", ""),
        "keyPoints": key_points_str,
        "questionCount": quiz_cfg.get("questionCount", 3),
        "difficulty": quiz_cfg.get("difficulty", "medium"),
        "questionTypes": ", ".join(quiz_cfg.get("questionTypes", ["single", "multiple"])),
    })
    if not prompts:
        raise ValueError("quiz-content template not found")

    raw = await call_llm(
        resolved,
        [{"role": "system", "content": prompts["system"]},
         {"role": "user",   "content": prompts["user"]}],
    )
    questions = _parse_json(raw)
    # quiz-content returns a JSON array of questions directly
    if isinstance(questions, list):
        return {"questions": questions}
    # Already wrapped
    if isinstance(questions, dict) and "questions" in questions:
        return questions
    return {"questions": []}


async def _gen_interactive_content(
    outline: dict[str, Any],
    language_directive: str,
    resolved,
) -> dict[str, Any]:
    # interactiveConfig from standard outline; widgetOutline from interactive-mode outline
    cfg = outline.get("interactiveConfig") or outline.get("widgetOutline") or {}
    key_points = outline.get("keyPoints", [])
    key_points_str = ", ".join(key_points) if isinstance(key_points, list) else str(key_points)

    # Derive language tag for the template
    lang_tag = "zh-CN"
    if language_directive and ("english" in language_directive.lower() or "en-US" in language_directive):
        lang_tag = "en-US"

    prompts = build_prompt("interactive-html", {
        "conceptName": cfg.get("conceptName", outline.get("title", "")),
        "subject": cfg.get("subject", ""),
        "conceptOverview": cfg.get("conceptOverview", outline.get("description", "")),
        "keyPoints": key_points_str,
        "scientificConstraints": cfg.get("scientificConstraints", ""),
        "designIdea": cfg.get("designIdea", cfg.get("challenge", "")),
        "language": lang_tag,
    })
    if not prompts:
        raise ValueError("interactive-html template not found")

    raw = await call_llm(
        resolved,
        [{"role": "system", "content": prompts["system"]},
         {"role": "user",   "content": prompts["user"]}],
        max_tokens=16384,
    )

    # The template asks LLM to "Return the complete HTML document directly",
    # so the response is raw HTML, NOT JSON. Wrap it ourselves.
    raw = raw.strip()
    # Strip any accidental markdown code fence
    raw = re.sub(r"^```(?:html)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw).strip()

    try:
        # In case some models ignore the instruction and wrap in JSON
        content = _parse_json(raw)
        if isinstance(content, str):
            content = {"html": content}
    except (json.JSONDecodeError, ValueError):
        # Normal case: raw HTML
        content = {"html": raw}

    if not content.get("html"):
        content["html"] = ""
    return content


async def _gen_pbl_content(
    outline: dict[str, Any],
    resolved,
) -> dict[str, Any]:
    """PBL has no standalone content template — return projectConfig from outline."""
    pbl_cfg = outline.get("pblConfig", {})
    return {
        "projectConfig": {
            "projectTopic": pbl_cfg.get("projectTopic", outline.get("title", "")),
            "projectDescription": pbl_cfg.get("projectDescription", outline.get("description", "")),
            "targetSkills": pbl_cfg.get("targetSkills", []),
            "issueCount": pbl_cfg.get("issueCount", 3),
            "language": pbl_cfg.get("language", "zh-CN"),
        }
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/generate/scene-content")
async def scene_content(body: SceneContentBody, request: Request):
    if not body.outline:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "outline is required")

    # outline.type is the canonical field (from scene-outlines-stream)
    scene_type = body.outline.get("type") or body.outline.get("scene_type") or "slide"
    language_directive = (
        body.language_directive
        or body.stage_info.get("language", "")
        or body.language
        or ""
    )

    model_str = request.headers.get("x-model") or get_settings().default_model
    client_key = request.headers.get("x-api-key", "")
    client_url = request.headers.get("x-base-url", "")

    if client_url:
        err = validate_url_for_ssrf(client_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
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

    rid = getattr(request.state, "request_id", "-")
    log.info(
        "Generating content: '%s' (%s) rid=%s [model=%s]",
        body.outline.get("title", ""),
        scene_type,
        rid,
        model_str,
    )

    try:
        if scene_type == "slide":
            content = await _gen_slide_content(body.outline, body.agents, language_directive, resolved)
        elif scene_type == "quiz":
            content = await _gen_quiz_content(body.outline, resolved)
        elif scene_type == "interactive":
            content = await _gen_interactive_content(body.outline, language_directive, resolved)
        elif scene_type == "pbl":
            content = await _gen_pbl_content(body.outline, resolved)
        else:
            log.warning("Unknown scene type '%s', falling back to slide", scene_type)
            content = await _gen_slide_content(body.outline, body.agents, language_directive, resolved)
    except Exception as exc:
        log.error("Scene content error rid=%s: %s", rid, exc)
        raise ApiException(
            ErrorCode.UPSTREAM_ERROR,
            f"Scene content generation failed: {exc}",
            status_code=502,
        ) from exc

    log.info("Content generated: '%s'", body.outline.get("title", ""))

    return api_success(
        {"content": content, "effectiveOutline": body.outline},
        request=request,
    )
