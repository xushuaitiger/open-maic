"""Generation-related Pydantic models (mirrors lib/types/generation.ts)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# User requirements
# ---------------------------------------------------------------------------

class UserRequirements(BaseModel):
    requirement: str
    language: Literal["zh-CN", "en-US"] = "zh-CN"
    pdf_content: str | None = None
    research_context: str | None = None


# ---------------------------------------------------------------------------
# Scene outline
# ---------------------------------------------------------------------------

class SceneOutline(BaseModel):
    id: str
    title: str
    description: str | None = None
    scene_type: Literal["slide", "quiz", "interactive", "pbl"] = "slide"
    slide_count: int | None = None
    image_descriptions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF types
# ---------------------------------------------------------------------------

class PdfImage(BaseModel):
    id: str
    data: str       # base64
    page: int | None = None
    description: str | None = None


class ImageMapping(BaseModel):
    mappings: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage / agent info
# ---------------------------------------------------------------------------

class AgentInfo(BaseModel):
    id: str
    name: str
    role: Literal["teacher", "assistant", "student"]
    persona: str | None = None
    avatar: str | None = None
    color: str | None = None


# ---------------------------------------------------------------------------
# Slide content
# ---------------------------------------------------------------------------

class SlideElement(BaseModel):
    type: str
    id: str
    content: Any = None
    style: dict[str, Any] = Field(default_factory=dict)


class Slide(BaseModel):
    id: str
    title: str
    elements: list[SlideElement] = Field(default_factory=list)
    speaker_notes: str | None = None


class GeneratedSlideContent(BaseModel):
    scene_type: Literal["slide"] = "slide"
    slides: list[Slide] = Field(default_factory=list)
    summary: str | None = None


# ---------------------------------------------------------------------------
# Quiz content
# ---------------------------------------------------------------------------

class QuizOption(BaseModel):
    id: str
    text: str
    is_correct: bool = False


class QuizQuestion(BaseModel):
    id: str
    type: Literal["single", "multiple", "text"]
    question: str
    options: list[QuizOption] = Field(default_factory=list)
    correct_answer: str | None = None
    explanation: str | None = None
    points: int = 1
    comment_prompt: str | None = None


class GeneratedQuizContent(BaseModel):
    scene_type: Literal["quiz"] = "quiz"
    questions: list[QuizQuestion] = Field(default_factory=list)
    summary: str | None = None


# ---------------------------------------------------------------------------
# Interactive content
# ---------------------------------------------------------------------------

class GeneratedInteractiveContent(BaseModel):
    scene_type: Literal["interactive"] = "interactive"
    html: str
    summary: str | None = None


# ---------------------------------------------------------------------------
# PBL content
# ---------------------------------------------------------------------------

class PBLIssue(BaseModel):
    id: str
    title: str
    description: str
    person_in_charge: str | None = None
    generated_questions: str | None = None


class PBLAgent(BaseModel):
    id: str
    name: str
    role: str
    avatar: str | None = None
    system_prompt: str


class GeneratedPBLContent(BaseModel):
    scene_type: Literal["pbl"] = "pbl"
    scenario: str
    issues: list[PBLIssue] = Field(default_factory=list)
    agents: list[PBLAgent] = Field(default_factory=list)
    summary: str | None = None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class SpeechAction(BaseModel):
    type: Literal["speech"] = "speech"
    id: str
    agent_id: str
    text: str
    audio_id: str | None = None


class Action(BaseModel):
    type: str
    id: str
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generate classroom input
# ---------------------------------------------------------------------------

class GenerateClassroomInput(BaseModel):
    requirement: str
    pdf_content: dict[str, Any] | None = None   # {text, images}
    language: str | None = None
    enable_web_search: bool = False
    enable_image_generation: bool = False
    enable_video_generation: bool = False
    enable_tts: bool = False
    agent_mode: Literal["default", "generate"] = "default"
    # Passed through from client request headers
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
