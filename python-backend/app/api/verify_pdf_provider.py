"""POST /api/verify-pdf-provider — connectivity check for PDF providers."""

import logging

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("verify_pdf_provider")
router = APIRouter()


class VerifyPDFProviderRequest(BaseModel):
    provider_id: str = Field(..., min_length=1, alias="providerId")
    api_key: str = Field(default="", alias="apiKey")
    base_url: str = Field(default="", alias="baseUrl")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/verify-pdf-provider")
async def verify_pdf_provider(body: VerifyPDFProviderRequest, request: Request):
    if body.base_url:
        err = validate_url_for_ssrf(body.base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved_url = body.base_url or cfg.resolve_pdf_base_url(body.provider_id)
    if not resolved_url:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "Base URL is required")

    resolved_key = body.api_key or cfg.resolve_pdf_api_key(body.provider_id, body.api_key)

    headers: dict[str, str] = {}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            resp = await client.get(resolved_url, headers=headers)
    except Exception as exc:
        msg = str(exc)
        if "ECONNREFUSED" in msg or "ConnectError" in msg:
            msg = "Cannot connect — is the service running?"
        raise ApiException(ErrorCode.UPSTREAM_ERROR, msg, status_code=502) from exc

    if 300 <= resp.status_code < 400:
        raise ApiException(ErrorCode.UPSTREAM_ERROR, "Redirects not allowed", status_code=502)

    return api_success(
        {"message": "Connection successful", "status": resp.status_code},
        request=request,
    )
