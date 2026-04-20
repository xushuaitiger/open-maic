"""
POST /api/pbl/chat     — PBL runtime chat (handles @question / @judge mentions)
POST /api/pbl/generate — Generate a full PBL project config via agentic loop

Mirrors app/api/pbl/chat/route.ts and the generate_pbl flow.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.providers.llm import call_llm, resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("pbl_chat")
router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────

class PblChatBody(BaseModel):
    message: str
    agent: dict[str, Any]
    current_issue: dict[str, Any] | None = Field(default=None, alias="currentIssue")
    recent_messages: list[dict[str, Any]] = Field(
        default_factory=list, alias="recentMessages"
    )
    user_role: str | None = Field(default=None, alias="userRole")
    agent_type: Literal["question", "judge"] | None = Field(default=None, alias="agentType")

    model: str | None = None
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class PblGenerateBody(BaseModel):
    project_topic: str = Field(alias="projectTopic")
    project_description: str | None = Field(default="", alias="projectDescription")
    target_skills: list[str] = Field(default_factory=list, alias="targetSkills")
    issue_count: int = Field(default=3, alias="issueCount", ge=1, le=10)
    language: str = "en-US"

    model: str | None = None
    api_key: str | None = Field(default=None, alias="apiKey")
    base_url: str | None = Field(default=None, alias="baseUrl")

    model_config = {"populate_by_name": True, "extra": "ignore"}


# ── /api/pbl/chat ───────────────────────────────────────────────────────────

@router.post("/pbl/chat")
async def pbl_chat(body: PblChatBody, request: Request):
    if not body.message.strip():
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "message is required")
    if not body.agent:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "agent is required")

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
        )

    issue_context = ""
    if body.current_issue:
        ci = body.current_issue
        issue_context = (
            "\n\n## Current Issue\n"
            f"Title: {ci.get('title', '')}\n"
            f"Description: {ci.get('description', '')}\n"
            f"Person in Charge: {ci.get('person_in_charge', '')}"
        )
        gq = ci.get("generated_questions", "")
        if gq:
            label = "Questions to Evaluate Against" if body.agent_type == "judge" else "Generated Questions"
            issue_context += f"\n\n{label}:\n{gq}"

    recent_context = ""
    if body.recent_messages:
        lines = "\n".join(
            f"{m.get('agent_name', 'Agent')}: {m.get('message', '')}"
            for m in body.recent_messages[-5:]
        )
        recent_context = f"\n\n## Recent Conversation\n{lines}"

    role_context = (
        f"\n\nThe student's role is: {body.user_role}" if body.user_role else ""
    )
    system_prompt = (
        f"{body.agent.get('system_prompt', '')}{issue_context}{recent_context}{role_context}"
    )

    try:
        response_text = await call_llm(
            resolved,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": body.message},
            ],
        )
    except Exception as exc:
        log.error("PBL chat error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, str(exc), status_code=502) from exc

    return api_success(
        {"message": response_text, "agentName": body.agent.get("name", "")},
        request=request,
    )


# ── /api/pbl/generate ───────────────────────────────────────────────────────

@router.post("/pbl/generate")
async def pbl_generate(body: PblGenerateBody, request: Request):
    if not body.project_topic.strip():
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "projectTopic is required")

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
        )

    from core.pbl.generate_pbl import GeneratePBLConfig, generate_pbl_content

    progress_log: list[str] = []
    try:
        pbl_config = await generate_pbl_content(
            GeneratePBLConfig(
                project_topic=body.project_topic,
                project_description=body.project_description or "",
                target_skills=body.target_skills,
                issue_count=body.issue_count,
                language=body.language,
            ),
            resolved_model=resolved,
            on_progress=lambda msg: progress_log.append(msg),
        )
    except Exception as exc:
        log.error("PBL generate error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc, exc_info=True)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, str(exc), status_code=502) from exc

    return api_success(
        {"config": pbl_config.to_dict(), "progress": progress_log},
        request=request,
    )
