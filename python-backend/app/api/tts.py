"""POST /api/generate/tts — synthesize speech audio from text."""

import base64
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.tts import TTSConfig, generate_tts
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("tts")
router = APIRouter()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    audio_id: str = Field(..., min_length=1, alias="audioId")
    tts_provider_id: str = Field(..., min_length=1, alias="ttsProviderId")
    tts_voice: str = Field(..., min_length=1, alias="ttsVoice")
    tts_speed: float = Field(default=1.0, alias="ttsSpeed")
    tts_api_key: str = Field(default="", alias="ttsApiKey")
    tts_base_url: str = Field(default="", alias="ttsBaseUrl")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/generate/tts")
async def generate_tts_endpoint(body: TTSRequest, request: Request):
    if body.tts_provider_id == "browser-native-tts":
        raise ApiException(
            ErrorCode.INVALID_REQUEST,
            "browser-native-tts must be handled client-side",
        )

    if body.tts_base_url:
        err = validate_url_for_ssrf(body.tts_base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    api_key = body.tts_api_key or cfg.resolve_tts_api_key(body.tts_provider_id, body.tts_api_key)
    base_url = body.tts_base_url or cfg.resolve_tts_base_url(body.tts_provider_id)

    try:
        result = await generate_tts(
            TTSConfig(
                provider_id=body.tts_provider_id,
                voice=body.tts_voice,
                api_key=api_key,
                base_url=base_url,
                speed=body.tts_speed,
            ),
            body.text,
        )
    except Exception as exc:
        log.error("TTS error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"TTS failed: {exc}", status_code=502) from exc

    return api_success(
        {
            "audioId": body.audio_id,
            "base64": base64.b64encode(result.audio).decode(),
            "format": result.format,
        },
        request=request,
    )
