"""
API-key / Base-URL resolution with a four-tier priority chain.

Priority (highest → lowest):
  1. Client-supplied value (request header / body)
  2. User-saved settings in the SQLite settings DB (PUT /api/settings)
  3. Server-level config  (server-providers.yml / environment variables)
  4. Hard-coded provider defaults (base URL only, never a key)

Why a separate module?
  Each existing route already calls ``cfg.resolve_api_key(provider_id, client_key)``.
  Adding tier-2 (DB lookup) inline in every route would require await everywhere
  and would scatter the logic.  This module provides async wrappers that slot
  naturally into routes that already do an await.

Usage::

    from core.providers.key_resolver import resolve_llm_key, resolve_llm_url

    api_key  = await resolve_llm_key(provider_id, client_key_from_header, cfg)
    base_url = await resolve_llm_url(provider_id, client_url_from_header, cfg)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

log = logging.getLogger("key_resolver")

# Cache the loaded settings blob (TTL: effectively per-process).
# We use a simple dict rather than functools.cache so we can invalidate it
# when the user calls PUT /api/settings.
_settings_cache: dict[str, Any] | None = None
_settings_cache_dirty: bool = True


async def _get_saved_settings(profile: str = "default") -> dict[str, Any]:
    """Load settings from DB (cached per process; invalidated on saves).

    Always returns a dict (never raises) so that a MySQL outage degrades
    gracefully to server-yml / env-var keys instead of crashing LLM routes.
    """
    global _settings_cache, _settings_cache_dirty
    if not _settings_cache_dirty and _settings_cache is not None:
        return _settings_cache
    try:
        from core.storage.settings_db import load_settings
        data = await load_settings(profile=profile, namespace="all")
        _settings_cache = data or {}
        _settings_cache_dirty = False
    except Exception as exc:
        # DB unavailable → fall through to yml/env keys silently
        log.debug("Could not load saved settings (DB may be down): %s", exc)
        _settings_cache = {}
    return _settings_cache


def invalidate_settings_cache() -> None:
    """Call this whenever PUT /api/settings succeeds."""
    global _settings_cache_dirty
    _settings_cache_dirty = True


def _get_from_saved(
    saved: dict[str, Any],
    config_key: str,        # e.g. "providersConfig"
    provider_id: str,
    field: str,             # e.g. "apiKey" or "baseUrl"
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

async def resolve_llm_key(
    provider_id: str,
    client_key: str,
    cfg: Any,               # ServerConfig
    *,
    profile: str = "default",
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "providersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_api_key(provider_id)


async def resolve_llm_url(
    provider_id: str,
    client_url: str,
    cfg: Any,
    *,
    profile: str = "default",
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "providersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_base_url(provider_id)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

async def resolve_tts_key(
    provider_id: str, client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "ttsProvidersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_tts_api_key(provider_id)


async def resolve_tts_url(
    provider_id: str, client_url: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "ttsProvidersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_tts_base_url(provider_id)


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

async def resolve_asr_key(
    provider_id: str, client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "asrProvidersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_asr_api_key(provider_id)


async def resolve_asr_url(
    provider_id: str, client_url: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "asrProvidersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_asr_base_url(provider_id)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

async def resolve_pdf_key(
    provider_id: str, client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "pdfProvidersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_pdf_api_key(provider_id)


async def resolve_pdf_url(
    provider_id: str, client_url: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "pdfProvidersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_pdf_base_url(provider_id)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

async def resolve_image_key(
    provider_id: str, client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "imageProvidersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_image_api_key(provider_id)


async def resolve_image_url(
    provider_id: str, client_url: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "imageProvidersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_image_base_url(provider_id)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

async def resolve_video_key(
    provider_id: str, client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "videoProvidersConfig", provider_id, "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_video_api_key(provider_id)


async def resolve_video_url(
    provider_id: str, client_url: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_url:
        return client_url
    saved = await _get_saved_settings(profile)
    db_url = _get_from_saved(saved, "videoProvidersConfig", provider_id, "baseUrl")
    if db_url:
        return db_url
    return cfg.resolve_video_base_url(provider_id)


# ---------------------------------------------------------------------------
# Convenience: async resolve_model (replaces the sync call in every route)
# ---------------------------------------------------------------------------

async def resolve_model_async(
    model_str: str,
    client_key: str,
    client_url: str,
    cfg: Any,
    *,
    profile: str = "default",
) -> Any:
    """Full 4-tier resolution then resolve_model.

    Replaces the pattern::
        resolved = resolve_model(model_str, body.api_key or "", body.base_url or "", cfg)

    With::
        resolved = await resolve_model_async(model_str, body.api_key or "", body.base_url or "", cfg)
    """
    from core.providers.llm import resolve_model  # lazy to avoid circular

    # Extract provider_id from "provider:model" format
    parts = model_str.split(":", 1) if model_str else []
    provider_id = parts[0] if parts else "openai"

    api_key = await resolve_llm_key(provider_id, client_key, cfg, profile=profile)
    base_url = await resolve_llm_url(provider_id, client_url, cfg, profile=profile)
    return resolve_model(model_str, api_key, base_url, cfg)


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------

async def resolve_web_search_key(
    client_key: str, cfg: Any, *, profile: str = "default"
) -> str:
    if client_key:
        return client_key
    saved = await _get_saved_settings(profile)
    db_key = _get_from_saved(saved, "webSearchProvidersConfig", "tavily", "apiKey")
    if db_key:
        return db_key
    return cfg.resolve_web_search_api_key()
