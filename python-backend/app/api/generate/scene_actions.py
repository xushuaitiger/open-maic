"""POST /api/generate/scene-actions — generate actions for a scene."""

import logging
from typing import Any

from fastapi import APIRouter, Request
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
    content: dict[str, Any]
    agents: list[dict[str, Any]] = Field(default_factory=list)
    previous_speeches: list[str] = Field(default_factory=list, alias="previousSpeeches")
    stage_info: dict[str, Any] = Field(default_factory=dict, alias="stageInfo")
    language: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/scene-actions")
async def scene_actions(body: SceneActionsBody, request: Request):
    if not body.outline:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "outline is required")
    if not body.content:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "content is required")

    language = _normalize_language(body.stage_info.get("language") or body.language)

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

    return api_success({"actions": actions}, request=request)
