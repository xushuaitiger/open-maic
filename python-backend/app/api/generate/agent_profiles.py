"""POST /api/generate/agent-profiles — auto-design teacher / assistant / student agents."""

import logging
from typing import Any

from fastapi import APIRouter, Request
from nanoid import generate as nanoid
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.generation.prompt_strings import AGENT_PROFILES_SYSTEM, agent_profiles_user_prompt
from core.providers.json_utils import parse_llm_json
from core.providers.llm import call_llm, resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("agent_profiles")
router = APIRouter()

_COLOR_PALETTE = [
    "#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#06b6d4",
    "#8b5cf6", "#f97316", "#14b8a6", "#e11d48", "#6366f1",
]


class AgentProfilesBody(BaseModel):
    stage_info: dict[str, Any] = Field(..., alias="stageInfo")
    scene_outlines: list[dict[str, Any]] = Field(default_factory=list, alias="sceneOutlines")
    language: str = "zh-CN"
    available_avatars: list[str] = Field(default_factory=list, alias="availableAvatars")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/agent-profiles")
async def agent_profiles(body: AgentProfilesBody, request: Request):
    if not body.stage_info.get("name"):
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "stageInfo.name is required")

    model_str = request.headers.get("x-model") or get_settings().default_model
    api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved = resolve_model(model_str, api_key, client_base_url, cfg)

    user_prompt = agent_profiles_user_prompt(
        body.stage_info, body.scene_outlines, body.language, body.available_avatars
    )

    try:
        raw = await call_llm(
            resolved,
            [
                {"role": "system", "content": AGENT_PROFILES_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
        )
        data = parse_llm_json(raw, expect="object")
    except Exception as exc:
        log.error("Agent profiles error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Agent profiles generation failed: {exc}", status_code=502) from exc

    agents_raw = data.get("agents", []) if isinstance(data, dict) else []
    if len(agents_raw) < 2:
        raise ApiException(
            ErrorCode.PARSE_FAILED,
            f"Expected at least 2 agents, got {len(agents_raw)}",
            status_code=502,
        )
    teachers = [a for a in agents_raw if a.get("role") == "teacher"]
    if len(teachers) != 1:
        raise ApiException(
            ErrorCode.PARSE_FAILED,
            f"Expected exactly 1 teacher, got {len(teachers)}",
            status_code=502,
        )

    agents = [
        {
            "id": f"gen-{nanoid(6)}",
            "name": a["name"],
            "role": a["role"],
            "persona": a.get("persona", ""),
            "avatar": a.get("avatar", ""),
            "color": _COLOR_PALETTE[i % len(_COLOR_PALETTE)],
        }
        for i, a in enumerate(agents_raw)
    ]

    return api_success({"agents": agents}, request=request)
