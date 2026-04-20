"""
Settings persistence API — server-side storage for provider API keys and
user preferences.

Why this exists
───────────────
The original Next.js project stores everything in browser localStorage (Zustand
persist). That works fine for a single device, but API keys disappear when you:
  • clear the browser cache
  • switch to another device / browser
  • deploy the frontend on a server and access it from multiple clients

This API gives the Python backend a persistent, encrypted home for those settings
so they survive beyond the browser session.

Security model
──────────────
  • API keys are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before being
    written to the SQLite file.  See core/storage/settings_db.py for the key
    rotation / derivation logic.
  • GET /api/settings returns the ACTUAL keys (needed so the frontend can restore
    them into the in-memory Zustand store and send them in request headers as
    usual).  Over a local loopback this is fine; in production use HTTPS.
  • GET /api/settings?masked=true returns a version safe to show in the UI
    (keys replaced with "••••abcd").

Endpoints
─────────
  GET    /api/settings                  Load saved settings (optionally masked)
  PUT    /api/settings                  Save / update settings
  DELETE /api/settings                  Clear saved settings
  GET    /api/settings/profiles         List available profiles

Shape of the settings payload
──────────────────────────────
Mirrors the Zustand SettingsState (lib/store/settings.ts) so the frontend can
dump its store state to the API as-is and restore it on page load.

  {
    "providersConfig": {
      "<providerId>": { "apiKey": "...", "baseUrl": "...", "models": [...], ... }
    },
    "ttsProvidersConfig":   { "<providerId>": { "apiKey": "...", ... } },
    "asrProvidersConfig":   { ... },
    "pdfProvidersConfig":   { ... },
    "imageProvidersConfig": { ... },
    "videoProvidersConfig": { ... },
    "webSearchProvidersConfig": { ... },
    // Optional non-provider preferences
    "providerId": "qwen",
    "modelId":    "qwen3-max",
    "ttsProviderId": "openai-tts",
    "ttsVoice":   "nova",
    "ttsSpeed":   1.0,
    "asrProviderId": "openai-whisper",
    "asrLanguage": "zh",
    "pdfProviderId": "unpdf",
    "imageProviderId": "seedream",
    "videoProviderId": "seedance",
    "webSearchProviderId": "tavily",
    "ttsEnabled": true,
    "asrEnabled": true,
    "imageGenerationEnabled": false,
    "videoGenerationEnabled": false
  }
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.errors import ApiException, ErrorCode, api_success
from core.providers.key_resolver import invalidate_settings_cache
from core.storage.settings_db import (
    delete_settings,
    list_profiles,
    load_settings,
    mask_api_keys,
    save_settings,
)

# MySQL-specific error type (resolved lazily so import is optional)
def _is_db_error(exc: Exception) -> bool:
    try:
        import pymysql.err  # aiomysql uses pymysql under the hood
        return isinstance(exc, (pymysql.err.OperationalError, pymysql.err.Error, OSError))
    except ImportError:
        return False

log = logging.getLogger("settings_api")
router = APIRouter()

_NS = "all"  # single namespace for the whole settings blob


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class SettingsPayload(BaseModel):
    """Loose wrapper — we accept whatever the Zustand store serialises."""

    # Provider key configs (the most important fields for this API)
    providers_config: dict[str, Any] | None = Field(
        default=None, alias="providersConfig"
    )
    tts_providers_config: dict[str, Any] | None = Field(
        default=None, alias="ttsProvidersConfig"
    )
    asr_providers_config: dict[str, Any] | None = Field(
        default=None, alias="asrProvidersConfig"
    )
    pdf_providers_config: dict[str, Any] | None = Field(
        default=None, alias="pdfProvidersConfig"
    )
    image_providers_config: dict[str, Any] | None = Field(
        default=None, alias="imageProvidersConfig"
    )
    video_providers_config: dict[str, Any] | None = Field(
        default=None, alias="videoProvidersConfig"
    )
    web_search_providers_config: dict[str, Any] | None = Field(
        default=None, alias="webSearchProvidersConfig"
    )

    # Active selections
    provider_id: str | None = Field(default=None, alias="providerId")
    model_id: str | None = Field(default=None, alias="modelId")
    tts_provider_id: str | None = Field(default=None, alias="ttsProviderId")
    tts_voice: str | None = Field(default=None, alias="ttsVoice")
    tts_speed: float | None = Field(default=None, alias="ttsSpeed")
    asr_provider_id: str | None = Field(default=None, alias="asrProviderId")
    asr_language: str | None = Field(default=None, alias="asrLanguage")
    pdf_provider_id: str | None = Field(default=None, alias="pdfProviderId")
    image_provider_id: str | None = Field(default=None, alias="imageProviderId")
    image_model_id: str | None = Field(default=None, alias="imageModelId")
    video_provider_id: str | None = Field(default=None, alias="videoProviderId")
    video_model_id: str | None = Field(default=None, alias="videoModelId")
    web_search_provider_id: str | None = Field(
        default=None, alias="webSearchProviderId"
    )

    # Feature toggles
    tts_enabled: bool | None = Field(default=None, alias="ttsEnabled")
    asr_enabled: bool | None = Field(default=None, alias="asrEnabled")
    image_generation_enabled: bool | None = Field(
        default=None, alias="imageGenerationEnabled"
    )
    video_generation_enabled: bool | None = Field(
        default=None, alias="videoGenerationEnabled"
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# GET /api/settings
# ---------------------------------------------------------------------------

@router.get("/settings")
async def get_settings_endpoint(
    request: Request,
    profile: str = Query(default="default"),
    masked: bool = Query(
        default=False,
        description="If true, API keys are replaced with '••••abcd' for safe UI display.",
    ),
):
    """Load saved settings. Returns null data if nothing has been saved yet."""
    try:
        data = await load_settings(profile=profile, namespace=_NS)
    except Exception as exc:
        if _is_db_error(exc):
            raise ApiException(
                ErrorCode.UPSTREAM_ERROR,
                "Settings database unavailable. Check MySQL connection.",
                status_code=503,
            ) from exc
        raise
    if data is None:
        return api_success(None, request=request)

    result = mask_api_keys(data) if masked else data
    return api_success(result, request=request)


# ---------------------------------------------------------------------------
# PUT /api/settings
# ---------------------------------------------------------------------------

@router.put("/settings")
async def put_settings_endpoint(
    body: SettingsPayload,
    request: Request,
    profile: str = Query(default="default"),
    merge: bool = Query(
        default=True,
        description=(
            "If true (default), merge with existing saved settings so you can "
            "update one provider without overwriting others.  "
            "If false, replace everything."
        ),
    ),
):
    """Save provider configurations (API keys are encrypted at rest)."""
    incoming = body.model_dump(by_alias=True, exclude_none=True)
    if not incoming:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "Request body is empty")

    try:
        if merge:
            existing = await load_settings(profile=profile, namespace=_NS) or {}
            _deep_merge(existing, incoming)
            payload = existing
        else:
            payload = incoming

        await save_settings(payload, profile=profile, namespace=_NS)
    except Exception as exc:
        if _is_db_error(exc):
            raise ApiException(
                ErrorCode.UPSTREAM_ERROR,
                "Settings database unavailable. Check MySQL connection.",
                status_code=503,
            ) from exc
        raise
    invalidate_settings_cache()  # flush the in-process key-resolution cache
    return api_success(
        {
            "profile": profile,
            "message": "Settings saved",
            "keyCount": _count_api_keys(payload),
        },
        request=request,
    )


# ---------------------------------------------------------------------------
# DELETE /api/settings
# ---------------------------------------------------------------------------

@router.delete("/settings")
async def delete_settings_endpoint(
    request: Request,
    profile: str = Query(default="default"),
):
    """Clear all saved settings for a profile."""
    try:
        deleted = await delete_settings(profile=profile)
    except Exception as exc:
        if _is_db_error(exc):
            raise ApiException(
                ErrorCode.UPSTREAM_ERROR,
                "Settings database unavailable. Check MySQL connection.",
                status_code=503,
            ) from exc
        raise
    return api_success(
        {"profile": profile, "deleted": deleted > 0},
        request=request,
    )


# ---------------------------------------------------------------------------
# GET /api/settings/profiles
# ---------------------------------------------------------------------------

@router.get("/settings/profiles")
async def get_profiles(request: Request):
    """List all profiles that have saved settings."""
    try:
        profiles = await list_profiles()
    except Exception as exc:
        if _is_db_error(exc):
            raise ApiException(
                ErrorCode.UPSTREAM_ERROR,
                "Settings database unavailable. Check MySQL connection.",
                status_code=503,
            ) from exc
        raise
    return api_success(profiles, request=request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place.

    Dict values are merged recursively; all other types are replaced.
    This lets the frontend send a partial update (e.g. just one provider's
    apiKey) without clobbering everything else.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _count_api_keys(obj: Any, _count: list[int] | None = None) -> int:
    """Count non-empty apiKey fields in a settings blob (for log/response)."""
    root = _count is None
    if root:
        _count = [0]
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ("apikey", "api_key") and isinstance(v, str) and v:
                _count[0] += 1
            else:
                _count_api_keys(v, _count)
    elif isinstance(obj, list):
        for item in obj:
            _count_api_keys(item, _count)
    return _count[0] if root else 0
