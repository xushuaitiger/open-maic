"""
Application configuration.

Priority (highest → lowest):
  1. Environment variables
  2. server-providers.yml
  3. defaults
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # Default model
    default_model: str = "openai:gpt-4o-mini"

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    google_api_key: str = ""
    google_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimax.chat/v1"
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    doubao_api_key: str = ""
    doubao_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    # TTS
    tts_openai_api_key: str = ""
    tts_openai_base_url: str = ""
    tts_azure_api_key: str = ""
    tts_azure_base_url: str = ""
    tts_glm_api_key: str = ""
    tts_qwen_api_key: str = ""

    # ASR
    asr_openai_api_key: str = ""
    asr_openai_base_url: str = ""
    asr_qwen_api_key: str = ""

    # PDF
    pdf_mineru_base_url: str = ""
    pdf_mineru_api_key: str = ""

    # Image
    image_seedream_api_key: str = ""
    image_qwen_image_api_key: str = ""
    image_nano_banana_api_key: str = ""

    # Video
    video_seedance_api_key: str = ""
    video_kling_api_key: str = ""
    video_veo_api_key: str = ""
    video_sora_api_key: str = ""

    # Web search
    tavily_api_key: str = ""

    # Storage
    classrooms_dir: str = ""
    # Root data directory; Fernet key file and classrooms live here.
    data_dir: str = "data"

    # MySQL 5.6+ connection settings for settings persistence.
    # Set these via environment variables or .env file:
    #   MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "openMAIC"
    mysql_pool_min: int = 1
    mysql_pool_max: int = 10

    # Optional: provide a pre-generated Fernet key for settings encryption.
    # If omitted, one is auto-generated and saved to <data_dir>/.settings_key.
    settings_encryption_key: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def classrooms_data_dir(self) -> Path:
        if self.classrooms_dir:
            return Path(self.classrooms_dir)
        return Path(__file__).parent.parent / "data" / "classrooms"


# ---------------------------------------------------------------------------
# YAML overlay (server-providers.yml)
# ---------------------------------------------------------------------------

_YAML_LLM_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "kimi": "kimi",
    "minimax": "minimax",
    "glm": "glm",
    "siliconflow": "siliconflow",
    "doubao": "doubao",
}

_YAML_TTS_MAP: dict[str, str] = {
    "openai-tts": "tts_openai",
    "azure-tts": "tts_azure",
    "glm-tts": "tts_glm",
    "qwen-tts": "tts_qwen",
}

_YAML_ASR_MAP: dict[str, str] = {
    "openai-whisper": "asr_openai",
    "qwen-asr": "asr_qwen",
}

_YAML_PDF_MAP: dict[str, str] = {
    "mineru": "pdf_mineru",
}

_YAML_IMAGE_MAP: dict[str, str] = {
    "seedream": "image_seedream",
    "qwen-image": "image_qwen_image",
    "nano-banana": "image_nano_banana",
}

_YAML_VIDEO_MAP: dict[str, str] = {
    "seedance": "video_seedance",
    "kling": "video_kling",
    "veo": "video_veo",
    "sora": "video_sora",
}

_YAML_WEB_SEARCH_MAP: dict[str, str] = {
    "tavily": "tavily",
}


class ProviderEntry:
    def __init__(self, api_key: str = "", base_url: str = "", models: list[str] | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.models = models or []


class ServerConfig:
    """Merged config from env vars + YAML, used by provider resolvers."""

    def __init__(self, settings: Settings, yaml_path: Path | None = None):
        self._settings = settings
        self._yaml: dict[str, Any] = {}

        if yaml_path and yaml_path.exists():
            with open(yaml_path) as f:
                self._yaml = yaml.safe_load(f) or {}

        self._llm = self._build_section("providers", _YAML_LLM_MAP, "llm")
        self._tts = self._build_section("tts", _YAML_TTS_MAP, "tts")
        self._asr = self._build_section("asr", _YAML_ASR_MAP, "asr")
        self._pdf = self._build_section("pdf", _YAML_PDF_MAP, "pdf")
        self._image = self._build_section("image", _YAML_IMAGE_MAP, "image")
        self._video = self._build_section("video", _YAML_VIDEO_MAP, "video")
        self._web_search = self._build_web_search()

    def _build_section(
        self,
        yaml_key: str,
        id_map: dict[str, str],
        kind: str,
    ) -> dict[str, ProviderEntry]:
        result: dict[str, ProviderEntry] = {}

        # Load from YAML first
        yaml_section: dict[str, Any] = self._yaml.get(yaml_key, {}) or {}
        for provider_id, entry in yaml_section.items():
            if isinstance(entry, dict) and entry.get("apiKey"):
                result[provider_id] = ProviderEntry(
                    api_key=entry["apiKey"],
                    base_url=entry.get("baseUrl", ""),
                    models=entry.get("models"),
                )

        # Overlay with env vars
        s = self._settings
        for provider_id, attr_prefix in id_map.items():
            api_key_attr = f"{attr_prefix}_api_key"
            base_url_attr = f"{attr_prefix}_base_url"
            env_key = getattr(s, api_key_attr, "")
            env_url = getattr(s, base_url_attr, "")

            if provider_id in result:
                if env_key:
                    result[provider_id].api_key = env_key
                if env_url:
                    result[provider_id].base_url = env_url
            elif env_key:
                result[provider_id] = ProviderEntry(api_key=env_key, base_url=env_url)

        return result

    def _build_web_search(self) -> dict[str, ProviderEntry]:
        result: dict[str, ProviderEntry] = {}
        yaml_section = self._yaml.get("web-search", {}) or {}
        for pid, entry in yaml_section.items():
            if isinstance(entry, dict) and entry.get("apiKey"):
                result[pid] = ProviderEntry(api_key=entry["apiKey"])

        env_key = self._settings.tavily_api_key
        if "tavily" in result:
            if env_key:
                result["tavily"].api_key = env_key
        elif env_key:
            result["tavily"] = ProviderEntry(api_key=env_key)
        return result

    # ── LLM ─────────────────────────────────────────────────────────────────

    def get_llm_providers(self) -> dict[str, dict]:
        return {
            pid: {"baseUrl": e.base_url, "models": e.models}
            for pid, e in self._llm.items()
        }

    def resolve_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._llm.get(provider_id, ProviderEntry()).api_key

    def resolve_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._llm.get(provider_id, ProviderEntry()).base_url

    # ── TTS ─────────────────────────────────────────────────────────────────

    def get_tts_providers(self) -> dict[str, dict]:
        return {pid: {"baseUrl": e.base_url} for pid, e in self._tts.items()}

    def resolve_tts_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._tts.get(provider_id, ProviderEntry()).api_key

    def resolve_tts_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._tts.get(provider_id, ProviderEntry()).base_url

    # ── ASR ─────────────────────────────────────────────────────────────────

    def get_asr_providers(self) -> dict[str, dict]:
        return {pid: {"baseUrl": e.base_url} for pid, e in self._asr.items()}

    def resolve_asr_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._asr.get(provider_id, ProviderEntry()).api_key

    def resolve_asr_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._asr.get(provider_id, ProviderEntry()).base_url

    # ── PDF ─────────────────────────────────────────────────────────────────

    def get_pdf_providers(self) -> dict[str, dict]:
        return {pid: {"baseUrl": e.base_url} for pid, e in self._pdf.items()}

    def resolve_pdf_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._pdf.get(provider_id, ProviderEntry()).api_key

    def resolve_pdf_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._pdf.get(provider_id, ProviderEntry()).base_url

    # ── Image ────────────────────────────────────────────────────────────────

    def get_image_providers(self) -> dict[str, dict]:
        return {pid: {} for pid in self._image}

    def resolve_image_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._image.get(provider_id, ProviderEntry()).api_key

    def resolve_image_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._image.get(provider_id, ProviderEntry()).base_url

    # ── Video ────────────────────────────────────────────────────────────────

    def get_video_providers(self) -> dict[str, dict]:
        return {pid: {} for pid in self._video}

    def resolve_video_api_key(self, provider_id: str, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._video.get(provider_id, ProviderEntry()).api_key

    def resolve_video_base_url(self, provider_id: str, client_url: str = "") -> str:
        if client_url:
            return client_url
        return self._video.get(provider_id, ProviderEntry()).base_url

    # ── Web Search ───────────────────────────────────────────────────────────

    def get_web_search_providers(self) -> dict[str, dict]:
        return {pid: {} for pid in self._web_search}

    def resolve_web_search_api_key(self, client_key: str = "") -> str:
        if client_key:
            return client_key
        return self._web_search.get("tavily", ProviderEntry()).api_key


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_server_config() -> ServerConfig:
    settings = get_settings()
    yaml_path = Path(__file__).parent.parent / "server-providers.yml"
    return ServerConfig(settings, yaml_path)
