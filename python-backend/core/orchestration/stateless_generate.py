"""
Stateless multi-agent generation entry point.

Runs the LangGraph director-agent graph and yields StatelessEvent dicts
for SSE streaming to the client.

Uses LangGraph's `stream_mode="custom"` so that each node can push
events via `get_stream_writer()` in real time — the client receives
`text_delta` chunks as the LLM generates them, not after the node
finishes. We also consume `stream_mode="updates"` to observe the
per-turn state diffs (e.g. agent_responses) that we accumulate into
the final `directorState` payload returned to the client.

Mirrors lib/orchestration/stateless-generate.ts (statelessGenerate).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from core.orchestration.director_graph import (
    create_orchestration_graph,
    build_initial_state,
)
from core.providers.llm import ResolvedModel

log = logging.getLogger("stateless_generate")


async def stateless_generate(
    request: dict,
    resolved_model: ResolvedModel,
) -> AsyncGenerator[dict, None]:
    """
    Run the orchestration graph and yield SSE events in real time.

    Each event is a dict: {type: str, data: dict}

    Event types:
      thinking        — director/agent loading indicator
      agent_start     — agent begins its turn
      text_delta      — spoken text chunk (streamed as generated)
      action          — whiteboard/slide action
      agent_end       — agent finished
      cue_user        — hand control back to student
      done            — session finished (includes updated directorState)
      error           — something went wrong
    """
    agent_ids = (request.get("config") or {}).get("agentIds", [])
    log.info(
        "Starting orchestration: agents=%s messages=%d model=%s",
        agent_ids,
        len(request.get("messages") or []),
        f"{resolved_model.provider_id}:{resolved_model.model_id}",
    )

    graph = create_orchestration_graph()
    initial_state = build_initial_state(request, resolved_model)

    total_actions = 0
    total_agents = 0
    agent_had_content = False

    # Accumulate agent turn summaries from state updates
    agent_turns: list[dict] = []
    new_whiteboard: list[dict] = []

    try:
        # stream_mode=["custom", "updates"] → each yielded item is
        # (mode, payload) where payload depends on the mode.
        async for mode, payload in graph.astream(
            initial_state,
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                event = payload
                if not isinstance(event, dict):
                    continue
                etype = event.get("type")
                edata = event.get("data") or {}

                if etype == "agent_start":
                    total_agents += 1
                    agent_had_content = False
                elif etype == "text_delta":
                    if edata.get("content"):
                        agent_had_content = True
                elif etype == "action":
                    total_actions += 1
                    agent_had_content = True

                yield event

            elif mode == "updates":
                # payload = { node_name: {field: value, ...} }
                if not isinstance(payload, dict):
                    continue
                for _node, updates in payload.items():
                    if not isinstance(updates, dict):
                        continue
                    for turn in updates.get("agent_responses", []) or []:
                        agent_turns.append(turn)
                    for wb in updates.get("whiteboard_ledger", []) or []:
                        new_whiteboard.append(wb)

    except Exception as exc:
        log.error("Orchestration error: %s", exc, exc_info=True)
        yield {"type": "error", "data": {"message": str(exc)}}
        return

    # Build updated directorState payload for the client to echo back next turn
    incoming = request.get("directorState") or request.get("director_state") or {}
    prev_responses = list(incoming.get("agentResponses") or incoming.get("agent_responses") or [])
    prev_ledger = list(incoming.get("whiteboardLedger") or incoming.get("whiteboard_ledger") or [])
    prev_turn_count = incoming.get("turnCount") or incoming.get("turn_count") or 0

    director_state = {
        "turnCount": prev_turn_count + (1 if agent_turns else 0),
        "agentResponses": prev_responses + agent_turns,
        "whiteboardLedger": prev_ledger + new_whiteboard,
    }

    yield {
        "type": "done",
        "data": {
            "totalActions": total_actions,
            "totalAgents": total_agents,
            "agentHadContent": agent_had_content,
            "directorState": director_state,
        },
    }

    log.info(
        "Orchestration complete: agents=%d actions=%d hadContent=%s turnCount=%d",
        total_agents, total_actions, agent_had_content, director_state["turnCount"],
    )
