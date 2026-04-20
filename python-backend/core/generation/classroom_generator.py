"""
Main classroom generation pipeline.

Flow:
  1. Resolve model & validate API key
  2. (Optional) Web search for research context
  3. Generate scene outlines via LLM
  4. For each outline: generate content + actions in parallel
  5. (Optional) Generate images / videos / TTS
  6. Persist classroom to storage
  7. Report progress at each step
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from nanoid import generate as nanoid

from app.config import get_server_config, get_settings
from core.generation.prompt_strings import (
    OUTLINE_SYSTEM,
    SLIDE_SYSTEM,
    QUIZ_SYSTEM,
    INTERACTIVE_SYSTEM,
    PBL_SYSTEM,
    ACTIONS_SYSTEM,
    outline_user_prompt,
    slide_user_prompt,
    quiz_user_prompt,
    interactive_user_prompt,
    pbl_user_prompt,
    actions_user_prompt,
)
from core.providers.llm import call_llm, resolve_model, ResolvedModel
from core.storage.classroom_store import persist_classroom
from core.web_search.tavily import search_with_tavily

ProgressCallback = Callable[[dict], Coroutine]

_DEFAULT_AGENTS = [
    {"id": "teacher-default", "name": "Alice", "role": "teacher", "persona": "Experienced, clear, encouraging teacher."},
    {"id": "student-default", "name": "Bob", "role": "student", "persona": "Curious, asks good questions."},
    {"id": "assistant-default", "name": "Carol", "role": "assistant", "persona": "Helpful teaching assistant."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _normalize_language(language: str | None) -> str:
    return "en-US" if language == "en-US" else "zh-CN"


async def _ai_call(resolved: ResolvedModel, system: str, user: str) -> str:
    return await call_llm(
        resolved,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=8192,
    )


async def _parse_json(text: str) -> Any:
    clean = _strip_code_fences(text)
    return json.loads(clean)


# ---------------------------------------------------------------------------
# Agent profile generation (server-side auto-generate)
# ---------------------------------------------------------------------------

async def _generate_agent_profiles(requirement: str, language: str, resolved: ResolvedModel) -> list[dict]:
    system = "You are an expert instructional designer. Generate agent profiles for a multi-agent classroom. Return ONLY valid JSON."
    user = f"""Generate 3-4 agent profiles for: {requirement}
Language: {language}
Rules: exactly 1 teacher, rest are assistant or student.
Return: {{"agents": [{{"name":"","role":"teacher|assistant|student","persona":""}}]}}"""

    try:
        raw = await _ai_call(resolved, system, user)
        data = await _parse_json(raw)
        agents = data.get("agents", [])
        if len(agents) < 2:
            raise ValueError("Too few agents")
        if sum(1 for a in agents if a.get("role") == "teacher") != 1:
            raise ValueError("Expected exactly 1 teacher")
        return [
            {"id": f"gen-{i}", "name": a["name"], "role": a["role"], "persona": a.get("persona", "")}
            for i, a in enumerate(agents)
        ]
    except Exception:
        return _DEFAULT_AGENTS


# ---------------------------------------------------------------------------
# Outline generation
# ---------------------------------------------------------------------------

async def _generate_outlines(
    requirement: str,
    language: str,
    agents: list[dict],
    resolved: ResolvedModel,
    research_context: str = "",
    pdf_text: str = "",
) -> list[dict]:
    teacher = next((a for a in agents if a.get("role") == "teacher"), agents[0])
    teacher_context = f"Teacher: {teacher['name']} — {teacher.get('persona', '')}"

    user = outline_user_prompt(
        requirement=requirement,
        language=language,
        teacher_context=teacher_context,
        research_context=research_context,
        pdf_text=pdf_text,
    )
    raw = await _ai_call(resolved, OUTLINE_SYSTEM, user)
    outlines = await _parse_json(raw)
    if not isinstance(outlines, list):
        outlines = outlines.get("outlines", [])

    # Ensure IDs
    for i, o in enumerate(outlines):
        if not o.get("id"):
            o["id"] = f"scene_{nanoid(8)}"
        if not o.get("scene_type"):
            o["scene_type"] = "slide"

    return outlines


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

async def _generate_scene_content(
    outline: dict,
    all_outlines: list[dict],
    stage_info: dict,
    agents: list[dict],
    language: str,
    resolved: ResolvedModel,
) -> dict:
    scene_type = outline.get("scene_type", "slide")

    if scene_type == "slide":
        user = slide_user_prompt(outline, all_outlines, stage_info, agents, language)
        raw = await _ai_call(resolved, SLIDE_SYSTEM, user)
    elif scene_type == "quiz":
        user = quiz_user_prompt(outline, language)
        raw = await _ai_call(resolved, QUIZ_SYSTEM, user)
    elif scene_type == "interactive":
        user = interactive_user_prompt(outline, language)
        raw = await _ai_call(resolved, INTERACTIVE_SYSTEM, user)
    elif scene_type == "pbl":
        user = pbl_user_prompt(outline, agents, language)
        raw = await _ai_call(resolved, PBL_SYSTEM, user)
    else:
        raise ValueError(f"Unknown scene_type: {scene_type}")

    return await _parse_json(raw)


async def _generate_scene_actions(
    outline: dict,
    content: dict,
    agents: list[dict],
    language: str,
    resolved: ResolvedModel,
    previous_speeches: list[str] | None = None,
) -> list[dict]:
    user = actions_user_prompt(outline, content, agents, language, previous_speeches)
    try:
        raw = await _ai_call(resolved, ACTIONS_SYSTEM, user)
        data = await _parse_json(raw)
        return data.get("actions", [])
    except Exception:
        return []


async def _generate_scene(
    outline: dict,
    all_outlines: list[dict],
    stage_info: dict,
    agents: list[dict],
    language: str,
    resolved: ResolvedModel,
    stage_id: str,
    order: int,
) -> dict:
    content = await _generate_scene_content(outline, all_outlines, stage_info, agents, language, resolved)
    previous_speeches = []
    actions = await _generate_scene_actions(outline, content, agents, language, resolved, previous_speeches)

    return {
        "id": outline["id"],
        "stage_id": stage_id,
        "title": outline.get("title", ""),
        "scene_type": outline.get("scene_type", "slide"),
        "content": content,
        "actions": actions,
        "outline": outline,
        "order": order,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def generate_classroom(
    input_data: dict,
    base_url: str,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """
    Full classroom generation pipeline.
    Returns a dict with {id, url, stage, scenes, scenes_count, created_at}.
    """

    async def progress(step: str, pct: int, message: str, scenes_done: int = 0, total: int | None = None):
        if on_progress:
            await on_progress({
                "step": step,
                "progress": pct,
                "message": message,
                "scenes_generated": scenes_done,
                "total_scenes": total,
            })

    await progress("initializing", 5, "Initializing classroom generation")

    cfg = get_server_config()
    settings = get_settings()
    model_string = input_data.get("model") or settings.default_model
    api_key = input_data.get("api_key", "")
    client_base_url = input_data.get("base_url", "")

    resolved = resolve_model(model_string, api_key, client_base_url, cfg)
    if not resolved.api_key:
        raise ValueError(
            f"No API key for provider '{resolved.provider_id}'. "
            f"Set {resolved.provider_id.upper()}_API_KEY in .env or server-providers.yml"
        )

    requirement: str = input_data.get("requirement", "")
    language = _normalize_language(input_data.get("language"))
    pdf_content = input_data.get("pdf_content") or {}
    pdf_text = pdf_content.get("text", "")

    # ── Agents ───────────────────────────────────────────────────────────────
    agent_mode = input_data.get("agent_mode", "default")
    if agent_mode == "generate":
        agents = await _generate_agent_profiles(requirement, language, resolved)
    else:
        agents = _DEFAULT_AGENTS

    # ── Web search ───────────────────────────────────────────────────────────
    await progress("researching", 10, "Researching topic")
    research_context = ""
    if input_data.get("enable_web_search"):
        tavily_key = cfg.resolve_web_search_api_key()
        if tavily_key:
            try:
                result = await search_with_tavily(requirement, tavily_key)
                research_context = result.context
            except Exception:
                pass  # graceful degradation

    # ── Outlines ─────────────────────────────────────────────────────────────
    await progress("generating_outlines", 20, "Generating course outlines")
    stage_id = nanoid(10)
    stage_info = {
        "id": stage_id,
        "name": requirement[:80],
        "description": requirement[:200],
        "language": language,
    }

    outlines = await _generate_outlines(
        requirement, language, agents, resolved, research_context, pdf_text
    )
    total_scenes = len(outlines)
    await progress("generating_scenes", 30, f"Generating {total_scenes} scenes", 0, total_scenes)

    # ── Scenes (parallel, max 3 concurrent) ──────────────────────────────────
    semaphore = asyncio.Semaphore(3)
    scenes: list[dict | None] = [None] * total_scenes
    completed = 0

    async def generate_with_sem(outline: dict, idx: int):
        nonlocal completed
        async with semaphore:
            scene = await _generate_scene(
                outline, outlines, stage_info, agents, language, resolved, stage_id, idx
            )
            scenes[idx] = scene
            completed += 1
            pct = 30 + int(completed / total_scenes * 50)
            await progress("generating_scenes", pct, f"Scene {completed}/{total_scenes} done", completed, total_scenes)

    await asyncio.gather(*[generate_with_sem(o, i) for i, o in enumerate(outlines)])
    final_scenes = [s for s in scenes if s is not None]

    # ── Persist ──────────────────────────────────────────────────────────────
    await progress("persisting", 90, "Saving classroom")
    now = datetime.now(timezone.utc).isoformat()
    stage = {**stage_info, "agents": agents, "created_at": now}
    classroom_data = {"id": stage_id, "stage": stage, "scenes": final_scenes}
    result = await persist_classroom(classroom_data, base_url)

    await progress("completed", 100, "Classroom generation complete", len(final_scenes), len(final_scenes))

    return {
        "id": result["id"],
        "url": result["url"],
        "stage": stage,
        "scenes": final_scenes,
        "scenes_count": len(final_scenes),
        "created_at": now,
    }
