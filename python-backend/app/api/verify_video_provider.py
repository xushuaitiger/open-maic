"""POST /api/verify-video-provider — connectivity check for video providers."""

import logging

from fastapi import APIRouter, Request

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.video import VideoConfig, test_video_connectivity
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("verify_video_provider")
router = APIRouter()


@router.post("/verify-video-provider")
async def verify_video_provider(request: Request):
    provider_id = request.headers.get("x-video-provider", "seedance")
    model = request.headers.get("x-video-model", "")
    client_api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    api_key = client_api_key or cfg.resolve_video_api_key(provider_id, client_api_key)
    base_url = client_base_url or cfg.resolve_video_base_url(provider_id)

    if not api_key:
        raise ApiException(ErrorCode.PROVIDER_ERROR, "No API key configured for this video provider")

    result = await test_video_connectivity(
        VideoConfig(provider_id=provider_id, api_key=api_key, base_url=base_url, model=model)
    )

    if not result.get("success"):
        raise ApiException(ErrorCode.UPSTREAM_ERROR, result.get("message", "Connection failed"), status_code=502)
    return api_success({"message": result["message"]}, request=request)
