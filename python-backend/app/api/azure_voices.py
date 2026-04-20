"""POST /api/azure-voices — fetch available Azure TTS voices."""

import logging

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.errors import ApiException, ErrorCode, api_success
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("azure_voices")
router = APIRouter()


class AzureVoicesRequest(BaseModel):
    api_key: str = Field(..., min_length=1, alias="apiKey")
    base_url: str = Field(..., min_length=1, alias="baseUrl")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/azure-voices")
async def azure_voices(body: AzureVoicesRequest, request: Request):
    err = validate_url_for_ssrf(body.base_url)
    if err:
        raise ApiException(ErrorCode.INVALID_URL, err)

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            resp = await client.get(
                f"{body.base_url}/cognitiveservices/voices/list",
                headers={"Ocp-Apim-Subscription-Key": body.api_key},
            )
    except Exception as exc:
        log.error("Azure voices error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, str(exc), status_code=502) from exc

    if 300 <= resp.status_code < 400:
        raise ApiException(ErrorCode.UPSTREAM_ERROR, "Redirects not allowed", status_code=502)
    if not resp.is_success:
        raise ApiException(
            ErrorCode.UPSTREAM_ERROR, f"Azure returned status {resp.status_code}", status_code=502
        )

    return api_success({"voices": resp.json()}, request=request)
