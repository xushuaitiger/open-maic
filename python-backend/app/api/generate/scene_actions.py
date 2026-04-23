"""POST /api/generate/scene-actions — generate actions for a scene."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from nanoid import generate as nanoid
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.generation.classroom_generator import _generate_scene_actions, _normalize_language
from core.providers.llm import resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("scene_actions")
router = APIRouter()


class SceneActionsBody(BaseModel):
    outline: dict[str, Any]
    all_outlines: list[dict[str, Any]] = Field(default_factory=list, alias="allOutlines")
    content: dict[str, Any]
    stage_id: str | None = Field(default=None, alias="stageId")
    agents: list[dict[str, Any]] = Field(default_factory=list)
    previous_speeches: list[str] = Field(default_factory=list, alias="previousSpeeches")
    user_profile: str | None = Field(default=None, alias="userProfile")
    language_directive: str | None = Field(default=None, alias="languageDirective")
    stage_info: dict[str, Any] = Field(default_factory=dict, alias="stageInfo")
    language: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


def _build_complete_scene(
    outline: dict[str, Any],
    content: dict[str, Any],
    actions: list[dict[str, Any]],
    stage_id: str,
) -> dict[str, Any] | None:
    """
    Mirrors TypeScript buildCompleteScene() from lib/generation/scene-builder.ts.
    Assembles outline + content + actions into a complete Scene object.
    """
    now = int(time.time() * 1000)
    scene_id = nanoid()
    scene_type = outline.get("type", "slide")

    base = {
        "id": scene_id,
        "stageId": stage_id,
        "type": scene_type,
        "title": outline.get("title", ""),
        "order": outline.get("order", 1),
        "actions": actions,
        "createdAt": now,
        "updatedAt": now,
    }

    if scene_type == "slide" and "elements" in content:
        default_theme = {
            "backgroundColor": "#ffffff",
            "themeColors": ["#5b9bd5", "#ed7d31", "#a5a5a5", "#ffc000", "#4472c4"],
            "fontColor": "#333333",
            "fontName": "Microsoft YaHei",
            "outline": {"color": "#d14424", "width": 2, "style": "solid"},
            "shadow": {"h": 0, "v": 0, "blur": 10, "color": "#000000"},
        }
        slide = {
            "id": nanoid(),
            "viewportSize": 1000,
            "viewportRatio": 0.5625,
            "theme": default_theme,
            "elements": content.get("elements", []),
            "background": content.get("background"),
        }
        return {**base, "content": {"type": "slide", "canvas": slide}}

    if scene_type == "quiz" and "questions" in content:
        return {**base, "content": {"type": "quiz", "questions": content.get("questions", [])}}

    if scene_type == "interactive" and ("html" in content or "widgetType" in content):
        return {
            **base,
            "content": {
                "type": "interactive",
                "url": "",
                "html": content.get("html", ""),
                "widgetType": content.get("widgetType"),
                "widgetConfig": content.get("widgetConfig"),
                "teacherActions": content.get("teacherActions"),
            },
        }

    if scene_type == "pbl" and "projectConfig" in content:
        return {**base, "content": {"type": "pbl", "projectConfig": content.get("projectConfig")}}

    # Fallback: wrap content as-is
    return {**base, "content": content}


@router.post("/generate/scene-actions")
async def scene_actions(body: SceneActionsBody, request: Request):
    if not body.outline:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "outline is required")
    if not body.content:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "content is required")

    language = _normalize_language(body.stage_info.get("language") or body.language)
    stage_id = body.stage_id or ""

    model_str = request.headers.get("x-model") or get_settings().default_model
    api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved = resolve_model(model_str, api_key, client_base_url, cfg)

    try:
        actions = await _generate_scene_actions(
            body.outline, body.content, body.agents, language, resolved, body.previous_speeches
        )
    except Exception as exc:
        log.error("Scene actions error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Scene actions generation failed: {exc}", status_code=502) from exc

    # Build complete scene (mirrors TS buildCompleteScene)
    scene = _build_complete_scene(body.outline, body.content, actions, stage_id)
    if not scene:
        raise ApiException(ErrorCode.GENERATION_FAILED, f"Failed to build scene: {body.outline.get('title', '')}", status_code=500)

    # Extract speech texts for cross-scene coherence (mirrors TS outputPreviousSpeeches)
    previous_speeches = [
        a.get("text", "")
        for a in actions
        if a.get("type") == "speech" and a.get("text")
    ]

    log.info(
        "Scene assembled: '%s' — %d actions",
        body.outline.get("title", ""),
        len(actions),
    )

    return api_success({"scene": scene, "previousSpeeches": previous_speeches}, request=request)
