"""POST /api/generate/scene-content — generate content for a single scene."""

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.generation.classroom_generator import _generate_scene_content, _normalize_language
from core.providers.llm import resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("scene_content")
router = APIRouter()


class SceneContentBody(BaseModel):
    outline: dict[str, Any]
    all_outlines: list[dict[str, Any]] = Field(default_factory=list, alias="allOutlines")
    stage_info: dict[str, Any] = Field(default_factory=dict, alias="stageInfo")
    agents: list[dict[str, Any]] = Field(default_factory=list)
    language: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/scene-content")
async def scene_content(body: SceneContentBody, request: Request):
    if not body.outline:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "outline is required")
    if not body.all_outlines:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "allOutlines is required")

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
        content = await _generate_scene_content(
            body.outline, body.all_outlines, body.stage_info, body.agents, language, resolved
        )
    except Exception as exc:
        log.error("Scene content error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Scene content generation failed: {exc}", status_code=502) from exc

    return api_success({"content": content}, request=request)
