"""POST /api/transcription — speech-to-text via configured ASR provider."""

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.asr import ASRConfig, transcribe_audio
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("transcription")
router = APIRouter()


@router.post("/transcription")
async def transcription(
    request: Request,
    audio: UploadFile = File(...),
    providerId: str = Form("openai-whisper"),
    language: str = Form("auto"),
    apiKey: str = Form(""),
    baseUrl: str = Form(""),
):
    if not audio or not audio.filename:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "audio file is required")

    if baseUrl:
        err = validate_url_for_ssrf(baseUrl)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved_key = apiKey or cfg.resolve_asr_api_key(providerId, apiKey)
    resolved_url = baseUrl or cfg.resolve_asr_base_url(providerId)

    try:
        audio_bytes = await audio.read()
        result = await transcribe_audio(
            ASRConfig(
                provider_id=providerId,
                api_key=resolved_key,
                base_url=resolved_url,
                language=language,
            ),
            audio_bytes,
            audio.filename or "audio.webm",
        )
    except Exception as exc:
        log.error("Transcription error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Transcription failed: {exc}", status_code=502) from exc

    return api_success({"text": result.text}, request=request)
