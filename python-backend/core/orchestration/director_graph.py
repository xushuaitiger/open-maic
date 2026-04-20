"""
Director Graph — LangGraph StateGraph for Multi-Agent Orchestration.

Topology (same as original TypeScript):

  START → director ──(end)──→ END
             │
             └─(next)→ agent_generate ──→ director (loop)

Director strategy:
  - Single agent: pure code logic (turn 0 → dispatch, turn 1+ → cue user)
  - Multi agent: LLM-based decision with code fast-paths

Mirrors lib/orchestration/director-graph.ts
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict, Annotated
import operator

from core.orchestration.agent_registry import (
    AgentConfig,
    get_agent,
    get_effective_actions,
)
from core.orchestration.prompt_builder import build_structured_prompt, build_director_prompt
from core.orchestration.stream_parser import (
    create_parser_state,
    parse_chunk,
    finalize_parser,
    ParsedAction,
    OrderedEntry,
)
from core.providers.llm import ResolvedModel, call_llm, stream_llm

log = logging.getLogger("director_graph")


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentTurnSummary(TypedDict):
    agent_id: str
    agent_name: str
    content_preview: str
    action_count: int
    whiteboard_actions: list[dict]


class OrchestratorState(TypedDict):
    # Input (set once)
    messages: list[dict]
    store_state: dict
    available_agent_ids: list[str]
    max_turns: int
    resolved_model: Any          # ResolvedModel (not serialisable, ok for in-process graph)
    discussion_context: dict | None
    trigger_agent_id: str | None
    user_profile: dict | None
    agent_config_overrides: dict  # agentId → raw dict
    # Mutable
    current_agent_id: str | None
    turn_count: int
    agent_responses: Annotated[list[AgentTurnSummary], operator.add]
    whiteboard_ledger: Annotated[list[dict], operator.add]
    should_end: bool
    total_actions: int
    # SSE event queue (each node appends events; the runner drains it)
    _events: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_agent(state: OrchestratorState, agent_id: str) -> AgentConfig | None:
    return get_agent(agent_id, state["agent_config_overrides"])


def _summarize_conversation(messages: list[dict]) -> str:
    lines = []
    for msg in messages[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {content[:120]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Director node
# ---------------------------------------------------------------------------

async def director_node(state: OrchestratorState) -> dict:
    events: list[dict] = []
    is_single = len(state["available_agent_ids"]) <= 1

    if state["turn_count"] >= state["max_turns"]:
        log.info("Director: turn limit reached, ending")
        return {"should_end": True, "_events": events}

    # Single agent: pure code
    if is_single:
        agent_id = state["available_agent_ids"][0] if state["available_agent_ids"] else "default-1"
        if state["turn_count"] == 0:
            events.append({"type": "thinking", "data": {"stage": "agent_loading", "agentId": agent_id}})
            return {"current_agent_id": agent_id, "should_end": False, "_events": events}
        else:
            events.append({"type": "cue_user", "data": {"fromAgentId": state["current_agent_id"]}})
            return {"should_end": True, "_events": events}

    # Multi-agent: fast-path for first turn with trigger
    trigger = state.get("trigger_agent_id")
    if state["turn_count"] == 0 and trigger and trigger in state["available_agent_ids"]:
        events.append({"type": "thinking", "data": {"stage": "agent_loading", "agentId": trigger}})
        return {"current_agent_id": trigger, "should_end": False, "_events": events}

    # Multi-agent: LLM decision
    agents = [_resolve_agent(state, aid) for aid in state["available_agent_ids"]]
    agents = [a for a in agents if a is not None]
    if not agents:
        return {"should_end": True, "_events": events}

    events.append({"type": "thinking", "data": {"stage": "director"}})

    conversation = _summarize_conversation(state["messages"])
    prompt = build_director_prompt(
        agents=agents,
        conversation_summary=conversation,
        agent_responses=state.get("agent_responses", []),
        turn_count=state["turn_count"],
        discussion_context=state.get("discussion_context"),
        trigger_agent_id=state.get("trigger_agent_id"),
        whiteboard_ledger=state.get("whiteboard_ledger", []),
        user_profile=state.get("user_profile"),
    )

    resolved: ResolvedModel = state["resolved_model"]
    try:
        decision_text = await call_llm(
            resolved,
            [{"role": "system", "content": prompt},
             {"role": "user", "content": "Decide which agent should speak next."}],
            max_tokens=16,
            temperature=0.2,
        )
        decision = decision_text.strip().strip('"').strip()

        if decision in ("END", "end") or not decision:
            return {"should_end": True, "_events": events}

        if decision == "USER":
            events.append({"type": "cue_user", "data": {"fromAgentId": state.get("current_agent_id")}})
            return {"should_end": True, "_events": events}

        if any(a.id == decision for a in agents):
            events.append({"type": "thinking", "data": {"stage": "agent_loading", "agentId": decision}})
            return {"current_agent_id": decision, "should_end": False, "_events": events}

        log.warning("Director: unknown agent '%s', ending", decision)
        return {"should_end": True, "_events": events}

    except Exception as exc:
        log.error("Director LLM error: %s", exc)
        return {"should_end": True, "_events": events}


def director_condition(state: OrchestratorState) -> str:
    return END if state["should_end"] else "agent_generate"


# ---------------------------------------------------------------------------
# Agent generate node
# ---------------------------------------------------------------------------

async def agent_generate_node(state: OrchestratorState) -> dict:
    agent_id = state.get("current_agent_id")
    if not agent_id:
        return {"should_end": True, "_events": []}

    agent = _resolve_agent(state, agent_id)
    if not agent:
        log.error("Agent not found: %s", agent_id)
        return {"should_end": True, "_events": []}

    events: list[dict] = []
    message_id = f"assistant-{agent_id}-{id(state)}"

    events.append({
        "type": "agent_start",
        "data": {
            "messageId": message_id,
            "agentId": agent_id,
            "agentName": agent.name,
            "agentAvatar": agent.avatar,
            "agentColor": agent.color,
        }
    })

    # Build effective actions based on current scene type
    scenes = state["store_state"].get("scenes", [])
    current_scene_id = state["store_state"].get("currentSceneId") or state["store_state"].get("current_scene_id")
    current_scene = next((s for s in scenes if s.get("id") == current_scene_id), None)
    scene_type = (current_scene or {}).get("type") or (current_scene or {}).get("scene_type")
    effective_actions = get_effective_actions(agent.allowed_actions, scene_type)

    system_prompt = build_structured_prompt(
        agent=agent,
        store_state=state["store_state"],
        discussion_context=state.get("discussion_context"),
        whiteboard_ledger=state.get("whiteboard_ledger", []),
        user_profile=state.get("user_profile"),
        agent_responses=state.get("agent_responses", []),
    )

    # Build LLM messages, mapping other agents' messages to "user" role
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in state["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # If message is from another agent (assistant but not this agent), treat as user
        if role == "assistant" and msg.get("agentId", "") != agent_id:
            role = "user"
        if isinstance(content, str):
            llm_messages.append({"role": role, "content": content})

    # Ensure we don't end with an assistant message
    if llm_messages and llm_messages[-1]["role"] == "assistant":
        llm_messages.append({"role": "user", "content": "It's your turn to speak."})
    if not any(m["role"] == "user" for m in llm_messages):
        llm_messages.append({"role": "user", "content": "Please begin."})

    resolved: ResolvedModel = state["resolved_model"]
    parser_state = create_parser_state()
    full_text = ""
    action_count = 0
    whiteboard_actions: list[dict] = []

    try:
        async for chunk in stream_llm(resolved, llm_messages):
            result = parse_chunk(chunk, parser_state)

            for entry in result.ordered:
                if entry.type == "text":
                    text = result.text_chunks[entry.index]
                    # Strip blockquote markers
                    text = "\n".join(
                        line.lstrip("> ").lstrip(">")
                        for line in text.split("\n")
                    )
                    if not text:
                        continue
                    full_text += text
                    events.append({
                        "type": "text_delta",
                        "data": {"content": text, "messageId": message_id}
                    })

                elif entry.type == "action":
                    action = result.actions[entry.index]
                    if action.action_name not in effective_actions:
                        log.warning("Agent %s attempted disallowed action: %s", agent.name, action.action_name)
                        continue
                    action_count += 1
                    if action.action_name.startswith("wb_"):
                        whiteboard_actions.append({
                            "actionName": action.action_name,
                            "agentId": agent_id,
                            "agentName": agent.name,
                            "params": action.params,
                        })
                    events.append({
                        "type": "action",
                        "data": {
                            "actionId": action.action_id,
                            "actionName": action.action_name,
                            "params": action.params,
                            "agentId": agent_id,
                            "messageId": message_id,
                        }
                    })

        # Finalize
        final = finalize_parser(parser_state)
        for entry in final.ordered:
            if entry.type == "text":
                text = final.text_chunks[entry.index]
                text = "\n".join(line.lstrip("> ").lstrip(">") for line in text.split("\n"))
                if text:
                    full_text += text
                    events.append({"type": "text_delta", "data": {"content": text, "messageId": message_id}})

    except Exception as exc:
        log.error("Agent generation error for %s: %s", agent.name, exc)
        events.append({"type": "error", "data": {"message": str(exc)}})

    events.append({"type": "agent_end", "data": {"messageId": message_id, "agentId": agent_id}})

    return {
        "turn_count": state["turn_count"] + 1,
        "total_actions": state.get("total_actions", 0) + action_count,
        "current_agent_id": None,
        "agent_responses": [{
            "agent_id": agent_id,
            "agent_name": agent.name,
            "content_preview": full_text[:300],
            "action_count": action_count,
            "whiteboard_actions": whiteboard_actions,
        }],
        "whiteboard_ledger": whiteboard_actions,
        "_events": events,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_orchestration_graph() -> CompiledStateGraph:
    """
    Build the LangGraph StateGraph.

    START → director ──(end)──→ END
               │
               └─(next)→ agent_generate ──→ director
    """
    graph = (
        StateGraph(OrchestratorState)
        .add_node("director", director_node)
        .add_node("agent_generate", agent_generate_node)
        .add_edge(START, "director")
        .add_conditional_edges("director", director_condition, {
            "agent_generate": "agent_generate",
            END: END,
        })
        .add_edge("agent_generate", "director")
    )
    return graph.compile()


def build_initial_state(
    request: dict,
    resolved_model: ResolvedModel,
) -> OrchestratorState:
    """Build initial graph state from a chat request dict."""
    config = request.get("config", {})
    agent_ids = config.get("agentIds") or config.get("agent_ids") or []

    # Build request-scoped overrides for generated agents
    overrides: dict = {}
    for cfg in (config.get("agentConfigs") or config.get("agent_configs") or []):
        overrides[cfg["id"]] = cfg

    discussion_context = None
    if config.get("discussionTopic") or config.get("discussion_topic"):
        discussion_context = {
            "topic": config.get("discussionTopic") or config.get("discussion_topic", ""),
            "prompt": config.get("discussionPrompt") or config.get("discussion_prompt", ""),
        }

    incoming = request.get("directorState") or request.get("director_state") or {}
    turn_count = incoming.get("turnCount") or incoming.get("turn_count") or 0

    return {
        "messages": request.get("messages", []),
        "store_state": request.get("storeState") or request.get("store_state") or {},
        "available_agent_ids": agent_ids,
        "max_turns": turn_count + 1,
        "resolved_model": resolved_model,
        "discussion_context": discussion_context,
        "trigger_agent_id": config.get("triggerAgentId") or config.get("trigger_agent_id"),
        "user_profile": request.get("userProfile") or request.get("user_profile"),
        "agent_config_overrides": overrides,
        "current_agent_id": None,
        "turn_count": turn_count,
        "agent_responses": list(incoming.get("agentResponses") or incoming.get("agent_responses") or []),
        "whiteboard_ledger": list(incoming.get("whiteboardLedger") or incoming.get("whiteboard_ledger") or []),
        "should_end": False,
        "total_actions": 0,
        "_events": [],
    }
