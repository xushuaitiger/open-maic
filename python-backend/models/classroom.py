"""Classroom / Stage / Scene Pydantic models (mirrors lib/types/stage.ts)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

class Stage(BaseModel):
    id: str
    name: str
    description: str | None = None
    language: Literal["zh-CN", "en-US"] = "zh-CN"
    agents: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class Scene(BaseModel):
    id: str
    stage_id: str
    title: str
    scene_type: Literal["slide", "quiz", "interactive", "pbl"] = "slide"
    content: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    outline: dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class Classroom(BaseModel):
    id: str
    stage: Stage
    scenes: list[Scene] = Field(default_factory=list)
    url: str | None = None
    created_at: str | None = None
