"""
TTS (Text-to-Speech) provider implementations.

Supported providers:
  - openai-tts  → OpenAI TTS API (model: gpt-4o-mini-tts)
  - azure-tts   → Azure Cognitive Services TTS (SSML)
  - glm-tts     → GLM TTS API
  - qwen-tts    → Qwen3 TTS Flash (DashScope)
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class TTSConfig:
    provider_id: str
    voice: str
    api_key: str
    base_url: str = ""
    speed: float = 1.0


@dataclass
class TTSResult:
    audio: bytes
    format: str  # "mp3" | "wav"


# ---------------------------------------------------------------------------
# Default base URLs
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URLS = {
    "openai-tts": "https://api.openai.com/v1",
    "glm-tts": "https://open.bigmodel.cn/api/paas/v4",
    "qwen-tts": "https://dashscope.aliyuncs.com/api/v1",
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

async def generate_tts(config: TTSConfig, text: str) -> TTSResult:
    match config.provider_id:
        case "openai-tts":
            return await _openai_tts(config, text)
        case "azure-tts":
            return await _azure_tts(config, text)
        case "glm-tts":
            return await _glm_tts(config, text)
        case "qwen-tts":
            return await _qwen_tts(config, text)
        case "browser-native-tts":
            raise ValueError("browser-native-tts must be handled client-side")
        case _:
            raise ValueError(f"Unknown TTS provider: {config.provider_id}")


# ---------------------------------------------------------------------------
# OpenAI TTS
# ---------------------------------------------------------------------------

async def _openai_tts(config: TTSConfig, text: str) -> TTSResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["openai-tts"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "model": "gpt-4o-mini-tts",
                "input": text,
                "voice": config.voice,
                "speed": config.speed,
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"OpenAI TTS error {resp.status_code}: {resp.text[:200]}")
    return TTSResult(audio=resp.content, format="mp3")


# ---------------------------------------------------------------------------
# Azure TTS
# ---------------------------------------------------------------------------

def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


async def _azure_tts(config: TTSConfig, text: str) -> TTSResult:
    base_url = config.base_url
    if not base_url:
        raise ValueError("Azure TTS requires a base_url (region endpoint)")

    rate_pct = int((config.speed - 1.0) * 100)
    rate_str = f"{rate_pct:+d}%"
    ssml = (
        f"<speak version='1.0' xml:lang='zh-CN'>"
        f"<voice xml:lang='zh-CN' name='{config.voice}'>"
        f"<prosody rate='{rate_str}'>{_escape_xml(text)}</prosody>"
        f"</voice></speak>"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/cognitiveservices/v1",
            headers={
                "Ocp-Apim-Subscription-Key": config.api_key,
                "Content-Type": "application/ssml+xml; charset=utf-8",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
            },
            content=ssml.encode("utf-8"),
        )
    if not resp.is_success:
        raise RuntimeError(f"Azure TTS error {resp.status_code}: {resp.text[:200]}")
    return TTSResult(audio=resp.content, format="mp3")


# ---------------------------------------------------------------------------
# GLM TTS
# ---------------------------------------------------------------------------

async def _glm_tts(config: TTSConfig, text: str) -> TTSResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["glm-tts"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "model": "glm-tts",
                "input": text,
                "voice": config.voice,
                "speed": config.speed,
                "volume": 1.0,
                "response_format": "wav",
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"GLM TTS error {resp.status_code}: {resp.text[:200]}")
    return TTSResult(audio=resp.content, format="wav")


# ---------------------------------------------------------------------------
# Qwen TTS
# ---------------------------------------------------------------------------

async def _qwen_tts(config: TTSConfig, text: str) -> TTSResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["qwen-tts"]
    rate = int((config.speed - 1.0) * 500)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "model": "qwen3-tts-flash",
                "input": {"text": text, "voice": config.voice, "language_type": "Chinese"},
                "parameters": {"rate": rate},
            },
        )

    if not resp.is_success:
        raise RuntimeError(f"Qwen TTS error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    audio_url = (data.get("output") or {}).get("audio", {}).get("url")
    if not audio_url:
        raise RuntimeError(f"Qwen TTS: no audio URL in response: {data}")

    async with httpx.AsyncClient(timeout=30) as client:
        audio_resp = await client.get(audio_url)
    if not audio_resp.is_success:
        raise RuntimeError(f"Qwen TTS download error {audio_resp.status_code}")

    return TTSResult(audio=audio_resp.content, format="wav")
