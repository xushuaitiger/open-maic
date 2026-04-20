"""POST /api/verify-model — test LLM connectivity."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.llm import call_llm, resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("verify_model")
router = APIRouter()


class VerifyModelRequest(BaseModel):
    model: str = Field(..., min_length=1)
    api_key: str = Field(default="", alias="apiKey")
    base_url: str = Field(default="", alias="baseUrl")
    provider_type: str = Field(default="", alias="providerType")
    requires_api_key: bool = Field(default=False, alias="requiresApiKey")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/verify-model")
async def verify_model(body: VerifyModelRequest, request: Request):
    if body.base_url:
        err = validate_url_for_ssrf(body.base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    try:
        resolved = resolve_model(body.model, body.api_key, body.base_url, cfg)
    except Exception as exc:
        raise ApiException(ErrorCode.INVALID_REQUEST, str(exc)) from exc

    try:
        text = await call_llm(
            resolved,
            [{"role": "user", "content": 'Say "OK" if you can hear me.'}],
            max_tokens=16,
        )
    except Exception as exc:
        raise ApiException(
            ErrorCode.UPSTREAM_ERROR,
            _friendly_error(str(exc)),
            status_code=502,
        ) from exc

    return api_success(
        {"message": "Connection successful", "response": text},
        request=request,
    )


def _friendly_error(msg: str) -> str:
    if "401" in msg or "unauthorized" in msg.lower():
        return "API key is invalid or expired"
    if "404" in msg or "not found" in msg.lower():
        return "Model not found or API endpoint error"
    if "429" in msg:
        return "API rate limit exceeded"
    if "ENOTFOUND" in msg or "ECONNREFUSED" in msg or "ConnectError" in msg:
        return "Cannot connect to API server — check the Base URL"
    if "timeout" in msg.lower():
        return "Connection timed out"
    return msg
