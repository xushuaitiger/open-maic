"""POST /api/generate/video — text-to-video generation."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.video import VideoConfig, VideoGenerationOptions, generate_video
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("video_gen")
router = APIRouter()


class VideoGenBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    duration: int = 5
    aspect_ratio: str = Field(default="16:9", alias="aspectRatio")
    resolution: str = "1080p"

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/video")
async def video_generation(body: VideoGenBody, request: Request):
    provider_id = request.headers.get("x-video-provider", "seedance")
    client_api_key = request.headers.get("x-api-key", "")
    client_base_url = request.headers.get("x-base-url", "")
    client_model = request.headers.get("x-video-model", "")

    if client_base_url:
        err = validate_url_for_ssrf(client_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    api_key = client_api_key or cfg.resolve_video_api_key(provider_id, client_api_key)
    base_url = client_base_url or cfg.resolve_video_base_url(provider_id)

    if not api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            f"No API key configured for video provider: {provider_id}",
        )

    try:
        result = await generate_video(
            VideoConfig(provider_id=provider_id, api_key=api_key, base_url=base_url, model=client_model),
            VideoGenerationOptions(
                prompt=body.prompt,
                duration=body.duration,
                aspect_ratio=body.aspect_ratio,
                resolution=body.resolution,
            ),
        )
    except Exception as exc:
        log.error("Video generation error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Video generation failed: {exc}", status_code=502) from exc

    return api_success(
        {"result": {"url": result.url, "format": result.format}},
        request=request,
    )
