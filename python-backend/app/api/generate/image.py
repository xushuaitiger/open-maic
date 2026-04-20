"""POST /api/generate/image — text-to-image generation."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.image import (
    ASPECT_RATIO_DIMENSIONS,
    ImageConfig,
    ImageGenerationOptions,
    generate_image,
)
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("image_gen")
router = APIRouter()


class ImageGenBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = Field(default="", alias="negativePrompt")
    width: int | None = None
    height: int | None = None
    aspect_ratio: str = Field(default="16:9", alias="aspectRatio")
    style: str = ""

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/image")
async def image_generation(body: ImageGenBody, request: Request):
    provider_id = request.headers.get("x-image-provider", "seedream")
    client_api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")
    client_model = request.headers.get("x-image-model", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    api_key = client_api_key or cfg.resolve_image_api_key(provider_id, client_api_key)
    base_url = client_base_url or cfg.resolve_image_base_url(provider_id)

    if not api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for image provider: {provider_id}",
        )

    w, h = ASPECT_RATIO_DIMENSIONS.get(body.aspect_ratio, (1024, 576))

    try:
        result = await generate_image(
            ImageConfig(provider_id=provider_id, api_key=api_key, base_url=base_url, model=client_model),
            ImageGenerationOptions(
                prompt=body.prompt,
                negative_prompt=body.negative_prompt,
                width=body.width or w,
                height=body.height or h,
                aspect_ratio=body.aspect_ratio,
                style=body.style,
            ),
        )
    except Exception as exc:
        log.error("Image generation error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Image generation failed: {exc}", status_code=502) from exc

    return api_success(
        {"result": {"url": result.url, "base64": result.base64, "format": result.format}},
        request=request,
    )
