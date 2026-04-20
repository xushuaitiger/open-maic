"""Chat-related models (mirrors lib/types/chat.ts)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[Any]
    id: str | None = None


class StoreState(BaseModel):
    stage: dict[str, Any] | None = None
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    current_scene_id: str | None = None
    mode: str = "playback"


class ChatConfig(BaseModel):
    agent_ids: list[str]
    session_type: str | None = None


class StatelessChatRequest(BaseModel):
    messages: list[ChatMessage]
    store_state: StoreState
    config: ChatConfig
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    thinking: dict[str, Any] | None = None
