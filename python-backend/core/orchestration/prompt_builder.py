"""
Build structured system prompts for agents.
Mirrors lib/orchestration/prompt-builder.ts
"""

from __future__ import annotations

from typing import Any

from core.orchestration.agent_registry import AgentConfig, get_action_descriptions


def build_structured_prompt(
    agent: AgentConfig,
    store_state: dict,
    discussion_context: dict | None = None,
    whiteboard_ledger: list[dict] | None = None,
    user_profile: dict | None = None,
    agent_responses: list[dict] | None = None,
) -> str:
    """
    Build the full system prompt for an agent turn.
    Output format: JSON array of {type:action|text} objects.
    """
    action_desc = get_action_descriptions(agent.allowed_actions)

    # Current scene context
    scenes = store_state.get("scenes", [])
    current_scene_id = store_state.get("currentSceneId") or store_state.get("current_scene_id")
    current_scene = next((s for s in scenes if s.get("id") == current_scene_id), None)
    scene_context = ""
    if current_scene:
        scene_type = current_scene.get("type") or current_scene.get("scene_type", "slide")
        scene_title = current_scene.get("title", "")
        scene_context = f"\n\n## Current Scene\nType: {scene_type}\nTitle: {scene_title}"
        if scene_type == "slide" and current_scene.get("content"):
            slides = current_scene["content"].get("slides", [])
            if slides:
                slide_titles = [s.get("title", "") for s in slides[:5]]
                scene_context += f"\nSlides: {', '.join(slide_titles)}"

    # Whiteboard state
    wb_context = ""
    if whiteboard_ledger:
        recent_wb = whiteboard_ledger[-5:]
        wb_lines = [f"- {w['agentName']} used {w['actionName']}" for w in recent_wb]
        wb_context = "\n\n## Recent Whiteboard Actions\n" + "\n".join(wb_lines)

    # Previous agent responses
    history_context = ""
    if agent_responses:
        recent = agent_responses[-3:]
        lines = [f"- {r['agentName']}: {r['contentPreview'][:80]}..." for r in recent if r.get("contentPreview")]
        if lines:
            history_context = "\n\n## Recent Turns\n" + "\n".join(lines)

    # Discussion context
    disc_context = ""
    if discussion_context:
        disc_context = f"\n\n## Discussion Topic\n{discussion_context.get('topic', '')}"
        if discussion_context.get("prompt"):
            disc_context += f"\n{discussion_context['prompt']}"

    # User profile
    user_ctx = ""
    if user_profile:
        name = user_profile.get("nickname", "")
        bio = user_profile.get("bio", "")
        if name or bio:
            user_ctx = f"\n\n## Student Profile\nName: {name}\n{bio}"

    return f"""{agent.persona}

## Your Role
Name: {agent.name}
Role: {agent.role}
{scene_context}{disc_context}{wb_context}{history_context}{user_ctx}

## Available Actions
{action_desc}

## Output Format
You MUST respond with a JSON array only — no other text outside the array.
Each item is either an action or a text chunk:

[
  {{"type": "action", "name": "spotlight", "params": {{"elementId": "img_1"}}}},
  {{"type": "text", "content": "Let me draw your attention to this diagram..."}},
  {{"type": "action", "name": "wb_open", "params": {{}}}},
  {{"type": "text", "content": "I'll write the formula on the board."}}
]

Rules:
- Text items contain your natural spoken words (what you actually say aloud)
- Action items are silent — never announce them in text ("I will now spotlight..." is wrong)
- Text and actions can freely interleave in the array
- Keep total spoken text to 50–150 words unless the topic requires more
- Always close the whiteboard (wb_close) after opening it
"""


def build_director_prompt(
    agents: list[AgentConfig],
    conversation_summary: str,
    agent_responses: list[dict],
    turn_count: int,
    discussion_context: dict | None = None,
    trigger_agent_id: str | None = None,
    whiteboard_ledger: list[dict] | None = None,
    user_profile: dict | None = None,
) -> str:
    agent_list = "\n".join(
        f"- id={a.id} name={a.name} role={a.role} priority={a.priority}"
        for a in agents
    )
    recent_turns = ""
    if agent_responses:
        recent_turns = "\n".join(
            f"- Turn {i+1}: {r['agentName']} ({r.get('actionCount', 0)} actions, preview: {r.get('contentPreview', '')[:60]}...)"
            for i, r in enumerate(agent_responses[-5:])
        )

    return f"""You are the director of a multi-agent AI classroom. Decide which agent should speak next.

## Available Agents
{agent_list}

## Conversation So Far
{conversation_summary}

## Recent Agent Turns
{recent_turns or "(none yet)"}

## Turn Info
Turn: {turn_count}

## Instructions
Respond with ONLY one of:
- An agent ID (e.g. "default-2") to dispatch that agent next
- "USER" to hand control back to the student
- "END" to finish the session

Choose based on:
1. Natural classroom flow (teacher usually goes first, then others chime in)
2. Avoid repeating the same agent consecutively
3. After 2-3 agent turns, consider handing to USER
4. For simple questions, 1-2 agents is enough

Decision:"""
