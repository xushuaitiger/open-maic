"""
API-key / Base-URL resolution with a three-tier priority chain.

Priority (highest → lowest):
  1. Client-supplied value (request header / body)
  2. Server-level config  (server-providers.yml / environment variables)
  3. Hard-coded provider defaults (base URL only, never a key)

User settings (API keys entered in the frontend settings panel) are stored in
the browser's localStorage via Zustand persist and sent directly in each
request header/body — so they arrive as tier-1 "client-supplied" values.
MySQL persistence for settings is reserved for a future iteration.

Usage::

    from core.providers.key_resolver import resolve_llm_key, resolve_llm_url

    api_key  = await resolve_llm_key(provider_id, client_key_from_header, cfg)
    base_url = await resolve_llm_url(provider_id, client_url_from_header, cfg)
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("key_resolver")


def _get_from_saved(
    saved: dict[str, Any],
    config_key: str,
    provider_id: str,
    field: str,
) -> str:
    """Drill into saved[config_key][provider_id][field], return "" if missing."""
    section = saved.get(config_key)
    if not isinstance(section, dict):
        return ""
    provider = section.get(provider_id)
    if not isinstance(provider, dict):
        return ""
    val = provider.get(field, "")
    return val if isinstance(val, str) else ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

async def resolve_llm_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_api_key(provider_id)


async def resolve_llm_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_base_url(provider_id)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

async def resolve_tts_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_tts_api_key(provider_id)


async def resolve_tts_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_tts_base_url(provider_id)


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

async def resolve_asr_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_asr_api_key(provider_id)


async def resolve_asr_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_asr_base_url(provider_id)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

async def resolve_pdf_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_pdf_api_key(provider_id)


async def resolve_pdf_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_pdf_base_url(provider_id)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

async def resolve_image_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_image_api_key(provider_id)


async def resolve_image_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_image_base_url(provider_id)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

async def resolve_video_key(provider_id: str, client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_video_api_key(provider_id)


async def resolve_video_url(provider_id: str, client_url: str, cfg: Any, **_: Any) -> str:
    return client_url or cfg.resolve_video_base_url(provider_id)


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------

async def resolve_web_search_key(client_key: str, cfg: Any, **_: Any) -> str:
    return client_key or cfg.resolve_web_search_api_key()


# ---------------------------------------------------------------------------
# Convenience: async resolve_model (wraps the sync call in routes)
# ---------------------------------------------------------------------------

async def resolve_model_async(
    model_str: str,
    client_key: str,
    client_url: str,
    cfg: Any,
    **_: Any,
) -> Any:
    """Two-tier resolution (client-supplied → server config) then resolve_model."""
    from core.providers.llm import resolve_model  # lazy to avoid circular

    parts = model_str.split(":", 1) if model_str else []
    provider_id = parts[0] if parts else "openai"
    api_key = await resolve_llm_key(provider_id, client_key, cfg)
    base_url = await resolve_llm_url(provider_id, client_url, cfg)
    return resolve_model(model_str, api_key, base_url, cfg)
