"""ASR (Automatic Speech Recognition) provider implementations."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class ASRConfig:
    provider_id: str
    api_key: str
    base_url: str = ""
    language: str = "auto"


@dataclass
class ASRResult:
    text: str


_DEFAULT_BASE_URLS = {
    "openai-whisper": "https://api.openai.com/v1",
    "qwen-asr": "https://dashscope.aliyuncs.com/api/v1",
}


async def transcribe_audio(config: ASRConfig, audio_bytes: bytes, filename: str = "audio.webm") -> ASRResult:
    match config.provider_id:
        case "openai-whisper":
            return await _openai_whisper(config, audio_bytes, filename)
        case "qwen-asr":
            return await _qwen_asr(config, audio_bytes, filename)
        case "browser-native":
            raise ValueError("browser-native ASR must be handled client-side")
        case _:
            raise ValueError(f"Unknown ASR provider: {config.provider_id}")


async def _openai_whisper(config: ASRConfig, audio_bytes: bytes, filename: str) -> ASRResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["openai-whisper"]
    files = {"file": (filename, audio_bytes, "application/octet-stream")}
    data: dict = {"model": "whisper-1"}
    if config.language and config.language != "auto":
        data["language"] = config.language

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            files=files,
            data=data,
        )
    if not resp.is_success:
        raise RuntimeError(f"Whisper error {resp.status_code}: {resp.text[:200]}")
    return ASRResult(text=resp.json().get("text", ""))


async def _qwen_asr(config: ASRConfig, audio_bytes: bytes, filename: str) -> ASRResult:
    import base64
    base_url = config.base_url or _DEFAULT_BASE_URLS["qwen-asr"]
    audio_b64 = base64.b64encode(audio_bytes).decode()

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "paraformer-realtime-v2",
                "input": {"audio": audio_b64},
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Qwen ASR error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    text = (data.get("output") or {}).get("text", "")
    return ASRResult(text=text)
