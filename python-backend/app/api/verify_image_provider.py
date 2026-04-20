"""POST /api/verify-image-provider — connectivity check for image providers."""

import logging

from fastapi import APIRouter, Request

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.image import ImageConfig, test_image_connectivity
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("verify_image_provider")
router = APIRouter()


@router.post("/verify-image-provider")
async def verify_image_provider(request: Request):
    provider_id = request.headers.get("x-image-provider", "seedream")
    model = request.headers.get("x-image-model", "")
    client_api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    api_key = client_api_key or cfg.resolve_image_api_key(provider_id, client_api_key)
    base_url = client_base_url or cfg.resolve_image_base_url(provider_id)

    if not api_key:
        raise ApiException(ErrorCode.PROVIDER_ERROR, "No API key configured for this image provider")

    result = await test_image_connectivity(
        ImageConfig(provider_id=provider_id, api_key=api_key, base_url=base_url, model=model)
    )

    if not result.get("success"):
        raise ApiException(ErrorCode.UPSTREAM_ERROR, result.get("message", "Connection failed"), status_code=502)
    return api_success({"message": result["message"]}, request=request)
