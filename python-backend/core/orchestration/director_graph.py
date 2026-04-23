"""
Director Graph — LangGraph StateGraph for Multi-Agent Orchestration.

Topology (same as original TypeScript):

  START → director ──(end)──→ END
             │
             └─(next)→ agent_generate ──→ director (loop)

Director strategy:
  - Single agent: pure code logic (turn 0 → dispatch, turn 1+ → cue user)
  - Multi agent: LLM-based decision with code fast-paths

Events are emitted in real time via `langgraph.config.get_stream_writer`
so the SSE client sees text deltas as they are generated — not buffered
until the node finishes.

Mirrors lib/orchestration/director-graph.ts
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict, Annotated
import operator

from core.orchestration.agent_registry import (
    AgentConfig,
    get_agent,
    get_effective_actions,
)
from core.orchestration.prompt_builder import (
    build_structured_prompt,
    build_director_prompt,
    summarize_conversation,
)
from core.orchestration.stream_parser import (
    create_parser_state,
    parse_chunk,
    finalize_parser,
)
from core.providers.llm import ResolvedModel, call_llm, stream_llm

log = logging.getLogger("director_graph")


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentTurnSummary(TypedDict):
    agentId: str
    agentName: str
    contentPreview: str
    actionCount: int
    whiteboardActions: list[dict]


class OrchestratorState(TypedDict):
    # Input (set once)
    messages: list[dict]
    store_state: dict
    available_agent_ids: list[str]
    max_turns: int
    resolved_model: Any
    discussion_context: dict | None
    trigger_agent_id: str | None
    user_profile: dict | None
    agent_config_overrides: dict
    # Mutable
    current_agent_id: str | None
    turn_count: int
    agent_responses: Annotated[list[AgentTurnSummary], operator.add]
    whiteboard_ledger: Annotated[list[dict], operator.add]
    should_end: bool
    total_actions: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_agent(state: OrchestratorState, agent_id: str) -> AgentConfig | None:
    return get_agent(agent_id, state["agent_config_overrides"])


def _safe_write(event: dict) -> None:
    """Write an event through the LangGraph stream writer; silently ignore if unavailable."""
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer is None:
        return
    try:
        writer(event)
    except Exception as exc:
        log.debug("stream writer rejected event: %s", exc)


# ---------------------------------------------------------------------------
# UIMessage (AI SDK v4) → OpenAI-style message conversion
# ---------------------------------------------------------------------------

def _text_from_part(part: Any) -> str | None:
    """Extract the text payload from a single UIMessage part, if any."""
    if not isinstance(part, dict):
        return None
    if part.get("type") == "text":
        text = part.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def _action_result_from_part(part: Any) -> tuple[str, str] | None:
    """Extract (action_name, result_summary) from an action-* part with a result."""
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    if not (isinstance(ptype, str) and ptype.startswith("action-")):
        return None
    if part.get("state") != "result":
        return None
    action_name = part.get("actionName") or ptype.replace("action-", "")
    output = part.get("output") or {}
    if output.get("success"):
        data = output.get("data")
        summary = (
            f"result: {json.dumps(data, ensure_ascii=False)[:100]}"
            if data is not None
            else "success"
        )
    else:
        summary = output.get("error") or "failed"
    return str(action_name), str(summary)


def _assistant_parts_to_json(parts: list[Any]) -> str:
    """Serialize assistant parts as the JSON array format the system prompt demands.

    This is what makes the model's OWN past turns serve as format few-shots —
    they appear in chat history in the exact `[{type,...}]` shape it's asked
    to emit. Without this, the model sees its prior responses as plain text
    and often drops the JSON structure on subsequent turns.

    Mirrors lib/orchestration/prompt-builder.ts — convertMessagesToOpenAI
    (assistant branch).
    """
    items: list[dict] = []
    for part in parts or []:
        text = _text_from_part(part)
        if text:
            items.append({"type": "text", "content": text})
            continue
        ar = _action_result_from_part(part)
        if ar:
            items.append({"type": "action", "name": ar[0], "result": ar[1]})
    return json.dumps(items, ensure_ascii=False) if items else ""


def _user_parts_to_text(parts: list[Any]) -> str:
    """Flatten user parts into plain text (text parts + action annotations)."""
    out: list[str] = []
    for part in parts or []:
        text = _text_from_part(part)
        if text:
            out.append(text)
            continue
        ar = _action_result_from_part(part)
        if ar:
            out.append(f"[Action {ar[0]}: {ar[1]}]")
    return "\n".join(out)


def _convert_messages(messages: list[dict], current_agent_id: str | None = None) -> list[dict]:
    """Convert UIMessage list to OpenAI chat.completions format.

    Mirrors lib/orchestration/prompt-builder.ts → convertMessagesToOpenAI:
      - Assistant messages serialize `parts` as a JSON array string so the
        model sees prior turns in the exact output format it is asked to
        emit.
      - User messages flatten `parts` to newline-joined plain text with
        `[Action <name>: <result>]` annotations and optional `[senderName]:`
        prefix.
      - When `current_agent_id` is provided, assistant messages from OTHER
        agents are rewritten as user messages with agent-name attribution,
        so the current agent treats them as external input, not its own
        history.
      - Legacy messages with a plain `content` string are also accepted.
    """
    converted: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
        sender_name = (metadata or {}).get("senderName")
        msg_agent_id = (metadata or {}).get("agentId") or msg.get("agentId")
        raw_parts = msg.get("parts")
        parts = raw_parts if isinstance(raw_parts, list) else None

        if role == "assistant":
            if parts is not None:
                content = _assistant_parts_to_json(parts)
            elif isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                content = _assistant_parts_to_json(msg["content"])
            else:
                content = ""

            # Assistant from a different agent → quote as user input
            if current_agent_id and msg_agent_id and msg_agent_id != current_agent_id:
                label = sender_name or msg_agent_id
                converted.append({
                    "role": "user",
                    "content": f"[{label}]: {content}" if content else "",
                })
                continue
            converted.append({"role": "assistant", "content": content})
            continue

        # user / system
        if parts is not None:
            text = _user_parts_to_text(parts)
        elif isinstance(msg.get("content"), str):
            text = msg["content"]
        elif isinstance(msg.get("content"), list):
            text = _user_parts_to_text(msg["content"])
        else:
            text = ""

        if role == "user" and sender_name:
            text = f"[{sender_name}]: {text}"
        if (metadata or {}).get("interrupted"):
            text = (
                f"{text}\n[This response was interrupted — do NOT continue it. "
                f"Start a new JSON array response.]"
            )
        converted.append({"role": role, "content": text})

    # Drop empty messages (matches TS .filter at end of convertMessagesToOpenAI)
    converted = [m for m in converted if (m.get("content") or "").strip()]
    return converted


_BLOCKQUOTE_RE = re.compile(r"^>+\s?", re.MULTILINE)


def _strip_blockquote_markers(text: str) -> str:
    """Strip leading `>` blockquote markers at line starts.

    Mirrors TS:  rawText.replace(/^>+\\s?/gm, '')
    Crucially does NOT strip leading whitespace in normal text — that would
    mangle tokens like ` Tiger` into `Tiger`.
    """
    return _BLOCKQUOTE_RE.sub("", text)


def _summarize_conversation(messages: list[dict]) -> str:
    """Summarize the conversation — delegated to the TS-aligned helper.

    First convert UIMessage parts → OpenAI format, then use the
    prompt-builder's summarize_conversation (same 10 messages × 200 chars
    trimming policy as the TypeScript version).
    """
    converted = _convert_messages(messages)
    return summarize_conversation(converted)


# ---------------------------------------------------------------------------
# Director node
# ---------------------------------------------------------------------------

async def director_node(state: OrchestratorState) -> dict:
    is_single = len(state["available_agent_ids"]) <= 1

    if state["turn_count"] >= state["max_turns"]:
        log.info("Director: turn limit reached (%d/%d)", state["turn_count"], state["max_turns"])
        return {"should_end": True}

    # Single agent: pure code
    if is_single:
        agent_id = state["available_agent_ids"][0] if state["available_agent_ids"] else "default-1"
        if state["turn_count"] == 0:
            log.info("Director[single]: dispatching %s", agent_id)
            _safe_write({"type": "thinking", "data": {"stage": "agent_loading", "agentId": agent_id}})
            return {"current_agent_id": agent_id, "should_end": False}
        log.info("Director[single]: cueing user after %s", state.get("current_agent_id"))
        _safe_write({"type": "cue_user", "data": {"fromAgentId": state.get("current_agent_id")}})
        return {"should_end": True}

    # Multi-agent: fast-path for first turn with trigger
    trigger = state.get("trigger_agent_id")
    if state["turn_count"] == 0 and trigger and trigger in state["available_agent_ids"]:
        log.info("Director[multi]: trigger fast-path → %s", trigger)
        _safe_write({"type": "thinking", "data": {"stage": "agent_loading", "agentId": trigger}})
        return {"current_agent_id": trigger, "should_end": False}

    # Multi-agent: LLM decision
    agents = [_resolve_agent(state, aid) for aid in state["available_agent_ids"]]
    agents = [a for a in agents if a is not None]
    if not agents:
        log.warning("Director: no available agents")
        return {"should_end": True}

    _safe_write({"type": "thinking", "data": {"stage": "director"}})

    conversation = _summarize_conversation(state["messages"])
    store_state = state.get("store_state") or {}
    prompt = build_director_prompt(
        agents=agents,
        conversation_summary=conversation,
        agent_responses=state.get("agent_responses", []),
        turn_count=state["turn_count"],
        discussion_context=state.get("discussion_context"),
        trigger_agent_id=state.get("trigger_agent_id"),
        whiteboard_ledger=state.get("whiteboard_ledger", []),
        user_profile=state.get("user_profile"),
        whiteboard_open=bool(store_state.get("whiteboardOpen")),
    )

    resolved: ResolvedModel = state["resolved_model"]
    log.info(
        "Director LLM call: turn=%d agents=%s responded=%d model=%s",
        state["turn_count"],
        [a.id for a in agents],
        len(state.get("agent_responses") or []),
        getattr(resolved, "model_id", "?"),
    )
    log.debug("Director conversation summary:\n%s", conversation)
    is_first_turn = state["turn_count"] == 0 and not (state.get("agent_responses") or [])

    def _force_first_agent(reason: str) -> dict:
        fallback = agents[0].id
        log.warning("Director: %s — forcing dispatch to first agent %s", reason, fallback)
        _safe_write({"type": "thinking", "data": {"stage": "agent_loading", "agentId": fallback}})
        return {"current_agent_id": fallback, "should_end": False}

    try:
        decision_text = await call_llm(
            resolved,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Decide which agent should speak next."},
            ],
            # Room for the model to produce a JSON object even if it emits
            # a short reasoning prefix. TS leaves this unbounded, but 512 is
            # comfortable for {"next_agent":"default-N"} + wrappers.
            max_tokens=512,
        )
        log.info("Director raw decision: %r", decision_text)

        decision = _extract_director_decision(decision_text, [a.id for a in agents])

        if decision in (None, "", "END", "end"):
            # Turn-0 END with zero agent responses is always a model mistake —
            # the user just spoke and SOMEONE needs to respond. Force the
            # highest-priority agent to speak instead of silently ending.
            if is_first_turn:
                return _force_first_agent(
                    f"first-turn decision {decision!r} but no agent has spoken"
                )
            log.info("Director decision: END")
            return {"should_end": True}

        if decision == "USER":
            # Same safety: cueing the user before any agent has spoken is wrong.
            if is_first_turn:
                return _force_first_agent("first-turn decision USER but no agent has spoken")
            log.info("Director decision: USER")
            _safe_write({"type": "cue_user", "data": {"fromAgentId": state.get("current_agent_id")}})
            return {"should_end": True}

        if any(a.id == decision for a in agents):
            log.info("Director decision: dispatch %s", decision)
            _safe_write({"type": "thinking", "data": {"stage": "agent_loading", "agentId": decision}})
            return {"current_agent_id": decision, "should_end": False}

        return _force_first_agent(f"unknown decision {decision!r}")

    except Exception as exc:
        log.error("Director LLM error: %s", exc, exc_info=True)
        if agents:
            return _force_first_agent(f"LLM error: {exc}")
        return {"should_end": True}


def _extract_director_decision(content: str, agent_ids: list[str]) -> str | None:
    """Extract a director decision from free-form LLM output.

    Accepts:
      - Plain agent ID: "default-1"
      - "USER" / "END" (case-insensitive)
      - JSON: {"next_agent": "default-1"}
      - Agent ID embedded in a sentence: "I choose default-1."
    """
    if not content:
        return None
    text = content.strip().strip('"').strip("`").strip()

    # Try JSON extraction first (mirrors TS parseDirectorDecision)
    json_match = re.search(r'\{[\s\S]*?"next_agent"[\s\S]*?\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            next_agent = parsed.get("next_agent")
            if next_agent:
                return str(next_agent).strip()
        except (json.JSONDecodeError, ValueError):
            pass

    upper = text.upper()
    if upper == "END" or upper.startswith("END "):
        return "END"
    if upper == "USER" or upper.startswith("USER "):
        return "USER"
    if text in agent_ids:
        return text

    for aid in agent_ids:
        if re.search(rf"\b{re.escape(aid)}\b", text):
            return aid

    if "END" in upper:
        return "END"
    if "USER" in upper:
        return "USER"
    return None


def director_condition(state: OrchestratorState) -> str:
    return END if state["should_end"] else "agent_generate"


# ---------------------------------------------------------------------------
# Agent generate node
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# text_delta coalescing
#
# LLMs often stream at token granularity (1-3 chars for Chinese models). One
# text_delta per LLM token makes the UI feel like it's typing one character
# at a time, because each SSE event carries a tiny payload + per-event round
# trip cost (Python → Starlette → Nginx → browser → React state update).
#
# We coalesce consecutive small text chunks into ~5-char bundles with a tiny
# max-wait time so the text still streams visibly in real time but each SSE
# event carries enough content to feel fluid.
# ---------------------------------------------------------------------------

_TEXT_COALESCE_MIN_CHARS = 4
_TEXT_COALESCE_MAX_WAIT_S = 0.04  # 40 ms — imperceptible to humans


class _TextBuffer:
    """Accumulate text deltas and flush in small bundles.

    Flushes when:
      - the accumulated length reaches ``_TEXT_COALESCE_MIN_CHARS``, OR
      - ``_TEXT_COALESCE_MAX_WAIT_S`` has elapsed since first-character
        buffered (caller drives time via ``flush_if_due``), OR
      - caller explicitly calls ``flush`` (end of turn, before action, etc.).
    """

    __slots__ = ("_buf", "_start_t", "_emit")

    def __init__(self, emit):
        self._buf: list[str] = []
        self._start_t: float | None = None
        self._emit = emit  # callable(str) → None

    def append(self, text: str, now: float) -> None:
        if not text:
            return
        if self._start_t is None:
            self._start_t = now
        self._buf.append(text)
        if sum(len(p) for p in self._buf) >= _TEXT_COALESCE_MIN_CHARS:
            self._flush_inner()

    def flush_if_due(self, now: float) -> None:
        if self._start_t is None or not self._buf:
            return
        if now - self._start_t >= _TEXT_COALESCE_MAX_WAIT_S:
            self._flush_inner()

    def flush(self) -> None:
        if self._buf:
            self._flush_inner()

    def _flush_inner(self) -> None:
        if not self._buf:
            return
        combined = "".join(self._buf)
        self._buf.clear()
        self._start_t = None
        if combined:
            self._emit(combined)


async def agent_generate_node(state: OrchestratorState) -> dict:
    agent_id = state.get("current_agent_id")
    if not agent_id:
        log.warning("agent_generate: no current_agent_id, ending")
        return {"should_end": True}

    agent = _resolve_agent(state, agent_id)
    if not agent:
        log.error("Agent not found: %s", agent_id)
        return {"should_end": True}

    import time
    message_id = f"assistant-{agent_id}-{int(time.time() * 1000)}"

    _safe_write({
        "type": "agent_start",
        "data": {
            "messageId": message_id,
            "agentId": agent_id,
            "agentName": agent.name,
            "agentAvatar": agent.avatar,
            "agentColor": agent.color,
        },
    })

    # Effective actions depend on the current scene type
    scenes = state["store_state"].get("scenes", [])
    current_scene_id = (
        state["store_state"].get("currentSceneId")
        or state["store_state"].get("current_scene_id")
    )
    current_scene = next((s for s in scenes if s.get("id") == current_scene_id), None)
    scene_type = (current_scene or {}).get("type") or (current_scene or {}).get("scene_type")
    effective_actions = get_effective_actions(agent.allowed_actions, scene_type)

    # Detect a direct user question from the last message so we can
    # surface it to the agent as high-priority. We only treat it as a
    # direct question if (a) the last message is user-role, and (b) its
    # parts contain real text (not just action-result annotations from
    # previous tool calls). This matches the sendMessage / QA flow.
    latest_user_question: str | None = None
    msgs = state.get("messages") or []
    if msgs:
        last = msgs[-1]
        if isinstance(last, dict) and last.get("role") == "user":
            parts = last.get("parts")
            text_fragments: list[str] = []
            if isinstance(parts, list):
                for p in parts:
                    t = _text_from_part(p)
                    if t and t.strip():
                        text_fragments.append(t.strip())
            elif isinstance(last.get("content"), str):
                text_fragments.append(last["content"].strip())
            combined = "\n".join(text_fragments).strip()
            # Skip generic internal prompts like "Please begin." that
            # the front-end never shows to the user.
            if combined and combined.lower() not in ("please begin.", "please begin"):
                latest_user_question = combined

    system_prompt = build_structured_prompt(
        agent=agent,
        store_state=state["store_state"],
        discussion_context=state.get("discussion_context"),
        whiteboard_ledger=state.get("whiteboard_ledger", []),
        user_profile=state.get("user_profile"),
        agent_responses=state.get("agent_responses", []),
        latest_user_question=latest_user_question,
    )

    # Convert UIMessages with agent-aware role remapping
    openai_messages = _convert_messages(state["messages"], current_agent_id=agent_id)

    llm_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(openai_messages)

    # Ensure the conversation has at least one user message and doesn't end on assistant
    if not any(m["role"] == "user" for m in llm_messages):
        llm_messages.append({"role": "user", "content": "Please begin."})
    elif llm_messages[-1]["role"] == "assistant":
        llm_messages.append(
            {"role": "user", "content": "It's your turn to speak. Respond from your perspective."}
        )

    last_user = next(
        (m for m in reversed(llm_messages) if m["role"] == "user"),
        None,
    )
    last_user_preview = ""
    if last_user:
        content = last_user.get("content") or ""
        last_user_preview = content[:120] + ("…" if len(content) > 120 else "")

    log.info(
        "Agent generate: agent=%s model=%s msg_count=%d last_user=%r",
        agent_id,
        getattr(state["resolved_model"], "model_id", "?"),
        len(llm_messages),
        last_user_preview,
    )

    resolved: ResolvedModel = state["resolved_model"]
    parser_state = create_parser_state()
    full_text = ""
    action_count = 0
    whiteboard_actions: list[dict] = []

    # Coalesce rapid text deltas so the UI doesn't have to process one SSE
    # event per Chinese character.  See _TextBuffer doc for details.
    def _emit_text(combined: str) -> None:
        _safe_write({
            "type": "text_delta",
            "data": {"content": combined, "messageId": message_id},
        })

    text_buffer = _TextBuffer(_emit_text)

    try:
        async for chunk in stream_llm(resolved, llm_messages):
            result = parse_chunk(chunk, parser_state)
            now = time.monotonic()

            for entry in result.ordered:
                if entry.type == "text":
                    text = result.text_chunks[entry.index]
                    # Strip blockquote markers at line starts, mirroring the TS
                    # regex /^>+\s?/gm — only removes `>` (and one optional whitespace),
                    # NOT leading spaces in normal text.
                    text = _strip_blockquote_markers(text)
                    if not text:
                        continue
                    full_text += text
                    text_buffer.append(text, now)

                elif entry.type == "action":
                    # Flush pending text before an action so ordering is preserved
                    text_buffer.flush()
                    action = result.actions[entry.index]
                    if action.action_name not in effective_actions:
                        log.warning(
                            "Agent %s attempted disallowed action: %s",
                            agent.name, action.action_name,
                        )
                        continue
                    action_count += 1
                    if action.action_name.startswith("wb_"):
                        whiteboard_actions.append({
                            "actionName": action.action_name,
                            "agentId": agent_id,
                            "agentName": agent.name,
                            "params": action.params,
                        })
                    _safe_write({
                        "type": "action",
                        "data": {
                            "actionId": action.action_id,
                            "actionName": action.action_name,
                            "params": action.params,
                            "agentId": agent_id,
                            "messageId": message_id,
                        },
                    })

            # Time-based flush: release accumulated text if it's been buffered
            # long enough, even if we haven't hit the char-count threshold.
            text_buffer.flush_if_due(now)

        # Finalize trailing content
        final = finalize_parser(parser_state)
        now = time.monotonic()
        for entry in final.ordered:
            if entry.type == "text":
                text = final.text_chunks[entry.index]
                text = _strip_blockquote_markers(text)
                if text:
                    full_text += text
                    text_buffer.append(text, now)

        # Always flush remaining text before the agent_end event
        text_buffer.flush()

    except Exception as exc:
        log.error("Agent generation error for %s: %s", agent.name, exc, exc_info=True)
        text_buffer.flush()
        _safe_write({"type": "error", "data": {"message": str(exc)}})

    _safe_write({"type": "agent_end", "data": {"messageId": message_id, "agentId": agent_id}})

    log.info(
        "Agent %s turn complete: text_len=%d actions=%d",
        agent_id, len(full_text), action_count,
    )

    return {
        "turn_count": state["turn_count"] + 1,
        "total_actions": state.get("total_actions", 0) + action_count,
        "current_agent_id": None,
        "agent_responses": [{
            "agentId": agent_id,
            "agentName": agent.name,
            "contentPreview": full_text[:100],  # matches TS slice(0, 100)
            "actionCount": action_count,
            "whiteboardActions": whiteboard_actions,
        }],
        "whiteboard_ledger": whiteboard_actions,
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
        .add_conditional_edges(
            "director",
            director_condition,
            {"agent_generate": "agent_generate", END: END},
        )
        .add_edge("agent_generate", "director")
    )
    return graph.compile()


def build_initial_state(
    request: dict,
    resolved_model: ResolvedModel,
) -> OrchestratorState:
    """Build initial graph state from a chat request dict."""
    config = request.get("config", {}) or {}
    agent_ids = config.get("agentIds") or config.get("agent_ids") or []

    overrides: dict = {}
    for cfg in (config.get("agentConfigs") or config.get("agent_configs") or []):
        if isinstance(cfg, dict) and cfg.get("id"):
            overrides[cfg["id"]] = cfg

    discussion_context = None
    topic = config.get("discussionTopic") or config.get("discussion_topic")
    if topic:
        discussion_context = {
            "topic": topic,
            "prompt": config.get("discussionPrompt") or config.get("discussion_prompt", ""),
        }

    incoming = request.get("directorState") or request.get("director_state") or {}
    turn_count = incoming.get("turnCount") or incoming.get("turn_count") or 0

    store_state = request.get("storeState") or request.get("store_state") or {}
    if not isinstance(store_state, dict):
        store_state = {}

    user_profile = request.get("userProfile") or request.get("user_profile")
    if not isinstance(user_profile, dict):
        user_profile = None

    return {
        "messages": request.get("messages", []) or [],
        "store_state": store_state,
        "available_agent_ids": agent_ids,
        "max_turns": turn_count + 1,
        "resolved_model": resolved_model,
        "discussion_context": discussion_context,
        "trigger_agent_id": config.get("triggerAgentId") or config.get("trigger_agent_id"),
        "user_profile": user_profile,
        "agent_config_overrides": overrides,
        "current_agent_id": None,
        "turn_count": turn_count,
        "agent_responses": list(incoming.get("agentResponses") or incoming.get("agent_responses") or []),
        "whiteboard_ledger": list(incoming.get("whiteboardLedger") or incoming.get("whiteboard_ledger") or []),
        "should_end": False,
        "total_actions": 0,
    }
