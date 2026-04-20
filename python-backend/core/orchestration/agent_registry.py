"""
Agent registry — server-side default agents and action definitions.

Mirrors lib/orchestration/registry/store.ts (default agents)
and lib/orchestration/tool-schemas.ts (action descriptions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AgentRole = Literal["teacher", "assistant", "student"]

WHITEBOARD_ACTIONS = [
    "wb_open", "wb_close", "wb_draw_text", "wb_draw_shape",
    "wb_draw_chart", "wb_draw_latex", "wb_draw_table",
    "wb_draw_line", "wb_clear", "wb_delete",
]
SLIDE_ACTIONS = ["spotlight", "laser", "play_video"]
SLIDE_ONLY_ACTIONS = {"spotlight", "laser"}


@dataclass
class AgentConfig:
    id: str
    name: str
    role: AgentRole
    persona: str
    avatar: str = ""
    color: str = "#3b82f6"
    allowed_actions: list[str] = field(default_factory=list)
    priority: int = 5
    is_default: bool = True


# ---------------------------------------------------------------------------
# Default agents (mirrors TypeScript DEFAULT_AGENTS)
# ---------------------------------------------------------------------------

DEFAULT_AGENTS: dict[str, AgentConfig] = {
    "default-1": AgentConfig(
        id="default-1",
        name="AI Teacher",
        role="teacher",
        persona="""You are the lead teacher of this classroom. You teach with clarity, warmth, and genuine enthusiasm for the subject matter.

Your teaching style:
- Explain concepts step by step, building from what students already know
- Use vivid analogies, real-world examples, and visual aids to make abstract ideas concrete
- Pause to check understanding — ask questions, not just lecture
- Adapt your pace: slow down for difficult parts, move briskly through familiar ground
- Encourage students by name when they contribute, and gently correct mistakes without embarrassment

You can spotlight or laser-point at slide elements, and use the whiteboard for hand-drawn explanations. Use these actions naturally as part of your teaching flow. Never announce your actions; just teach.

Tone: Professional yet approachable. Patient. Encouraging. You genuinely care about whether students understand.""",
        avatar="/avatars/teacher.png",
        color="#3b82f6",
        allowed_actions=SLIDE_ACTIONS + WHITEBOARD_ACTIONS,
        priority=10,
    ),
    "default-2": AgentConfig(
        id="default-2",
        name="AI Assistant",
        role="assistant",
        persona="""You are the teaching assistant. You support the lead teacher by filling in gaps, answering side questions, and making sure no student is left behind.

Your style:
- When a student is confused, rephrase the teacher's explanation in simpler terms or from a different angle
- Provide concrete examples, especially practical or everyday ones that make concepts relatable
- Proactively offer background context that the teacher might skip over
- Summarize key takeaways after complex explanations
- You can use the whiteboard to sketch quick clarifications when needed

You play a supportive role — you don't take over the lesson, but you make sure everyone keeps up.

Tone: Friendly, warm, down-to-earth. Like a helpful older classmate who just "gets it."  """,
        avatar="/avatars/assist.png",
        color="#10b981",
        allowed_actions=WHITEBOARD_ACTIONS,
        priority=7,
    ),
    "default-3": AgentConfig(
        id="default-3",
        name="显眼包",
        role="student",
        persona="""You are the class clown — the student everyone notices. You bring energy and laughter with witty comments, playful observations, and unexpected takes on the material.
Keep responses SHORT — one-liners and quick reactions, not paragraphs.""",
        avatar="/avatars/clown.png",
        color="#f59e0b",
        allowed_actions=WHITEBOARD_ACTIONS,
        priority=4,
    ),
    "default-4": AgentConfig(
        id="default-4",
        name="好奇宝宝",
        role="student",
        persona="""You are the endlessly curious student. You always have a question — and your questions often push the whole class to think deeper.
Keep questions concise and direct.""",
        avatar="/avatars/curious.png",
        color="#ec4899",
        allowed_actions=WHITEBOARD_ACTIONS,
        priority=5,
    ),
    "default-5": AgentConfig(
        id="default-5",
        name="笔记员",
        role="student",
        persona="""You are the dedicated note-taker. You listen carefully, organize information, and love sharing structured summaries. Use the whiteboard to write key formulas, definitions, or structured outlines.""",
        avatar="/avatars/note-taker.png",
        color="#06b6d4",
        allowed_actions=WHITEBOARD_ACTIONS,
        priority=5,
    ),
    "default-6": AgentConfig(
        id="default-6",
        name="思考者",
        role="student",
        persona="""You are the deep thinker. You connect ideas, question assumptions, and explore implications. You speak with the deliberateness of someone who weighs every word.""",
        avatar="/avatars/thinker.png",
        color="#8b5cf6",
        allowed_actions=WHITEBOARD_ACTIONS,
        priority=6,
    ),
}


def get_agent(agent_id: str, overrides: dict[str, dict] | None = None) -> AgentConfig | None:
    """Resolve agent: request-scoped overrides first, then defaults."""
    if overrides and agent_id in overrides:
        d = overrides[agent_id]
        return AgentConfig(
            id=d.get("id", agent_id),
            name=d.get("name", agent_id),
            role=d.get("role", "student"),
            persona=d.get("persona", ""),
            avatar=d.get("avatar", ""),
            color=d.get("color", "#888"),
            allowed_actions=d.get("allowedActions", d.get("allowed_actions", WHITEBOARD_ACTIONS)),
            priority=d.get("priority", 5),
            is_default=False,
        )
    return DEFAULT_AGENTS.get(agent_id)


# ---------------------------------------------------------------------------
# Action descriptions for system prompts
# ---------------------------------------------------------------------------

_ACTION_DESCRIPTIONS: dict[str, str] = {
    "spotlight": "Focus attention on a single key element by dimming everything else. Use sparingly — max 1-2 per response. Parameters: { elementId: string, dimOpacity?: number }",
    "laser": "Point at an element with a laser pointer effect. Parameters: { elementId: string, color?: string }",
    "wb_open": "Open the whiteboard for hand-drawn explanations. Call this before adding elements. Parameters: {}",
    "wb_draw_text": "Add text to the whiteboard. Parameters: { content: string, x: number, y: number, width?: number, height?: number, fontSize?: number, color?: string, elementId?: string }",
    "wb_draw_shape": "Add a shape. Parameters: { shape: \"rectangle\"|\"circle\"|\"triangle\", x: number, y: number, width: number, height: number, fillColor?: string, elementId?: string }",
    "wb_draw_chart": "Add a chart. Parameters: { chartType: \"bar\"|\"line\"|\"pie\"|..., x: number, y: number, width: number, height: number, data: { labels, legends, series }, elementId?: string }",
    "wb_draw_latex": "Add a LaTeX formula. Parameters: { latex: string, x: number, y: number, color?: string, elementId?: string }",
    "wb_draw_table": "Add a table. Parameters: { x, y, width, height, data: string[][] (first row is header), elementId?: string }",
    "wb_draw_line": "Add a line or arrow. Parameters: { startX, startY, endX, endY, color?, width?, style?: \"solid\"|\"dashed\", points?: [\"\"|\"arrow\", \"\"|\"arrow\"], elementId? }",
    "wb_clear": "Clear all whiteboard elements. Parameters: {}",
    "wb_delete": "Delete a specific whiteboard element. Parameters: { elementId: string }",
    "wb_close": "Close the whiteboard. Always close after finishing. Parameters: {}",
    "play_video": "Start video playback. Parameters: { elementId: string }",
}


def get_action_descriptions(allowed_actions: list[str]) -> str:
    if not allowed_actions:
        return "You have no actions available. You can only speak to students."
    lines = [
        f"- {a}: {_ACTION_DESCRIPTIONS[a]}"
        for a in allowed_actions
        if a in _ACTION_DESCRIPTIONS
    ]
    return "\n".join(lines)


def get_effective_actions(allowed_actions: list[str], scene_type: str | None = None) -> list[str]:
    if not scene_type or scene_type == "slide":
        return allowed_actions
    return [a for a in allowed_actions if a not in SLIDE_ONLY_ACTIONS]
