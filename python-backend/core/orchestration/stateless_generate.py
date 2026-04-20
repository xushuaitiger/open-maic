"""
Stateless multi-agent generation entry point.

Runs the LangGraph director-agent graph and yields StatelessEvent dicts
for SSE streaming to the client.

Mirrors lib/orchestration/stateless-generate.ts (statelessGenerate).
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

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
    Run the orchestration graph and yield SSE events.

    Each event is a dict: {type: str, data: dict}

    Event types:
      thinking        — director/agent loading indicator
      agent_start     — agent begins its turn
      text_delta      — spoken text chunk
      action          — whiteboard/slide action
      agent_end       — agent finished
      cue_user        — hand control back to student
      done            — session finished (includes updated directorState)
      error           — something went wrong
    """
    log.info(
        "Starting orchestration for agents: %s",
        request.get("config", {}).get("agentIds", []),
    )

    graph = create_orchestration_graph()
    initial_state = build_initial_state(request, resolved_model)

    total_actions = 0
    total_agents = 0
    agent_had_content = False
    current_agent_id = None
    current_agent_name = None
    content_preview = ""
    agent_action_count = 0
    agent_wb_actions: list[dict] = []

    try:
        # LangGraph streams the full state after each node.
        # We collect events from the `_events` field that each node appends.
        async for node_output in graph.astream(initial_state):
            # node_output is {node_name: updated_state_fields}
            for node_name, updates in node_output.items():
                if not isinstance(updates, dict):
                    continue
                new_events: list[dict] = updates.get("_events", [])
                for event in new_events:
                    etype = event.get("type")
                    edata = event.get("data", {})

                    if etype == "agent_start":
                        total_agents += 1
                        current_agent_id = edata.get("agentId")
                        current_agent_name = edata.get("agentName")
                        content_preview = ""
                        agent_action_count = 0
                        agent_wb_actions.clear()

                    if etype == "text_delta" and len(content_preview) < 100:
                        content_preview = (content_preview + edata.get("content", ""))[:100]
                        agent_had_content = True

                    if etype == "action":
                        total_actions += 1
                        agent_action_count += 1
                        agent_had_content = True
                        if edata.get("actionName", "").startswith("wb_"):
                            agent_wb_actions.append({
                                "actionName": edata["actionName"],
                                "agentId": edata.get("agentId", ""),
                                "agentName": current_agent_name or "",
                                "params": edata.get("params", {}),
                            })

                    yield event

    except Exception as exc:
        log.error("Orchestration error: %s", exc, exc_info=True)
        yield {"type": "error", "data": {"message": str(exc)}}
        return

    # Build updated directorState for the client to pass back next turn
    incoming = request.get("directorState") or request.get("director_state") or {}
    prev_responses = list(incoming.get("agentResponses") or incoming.get("agent_responses") or [])
    prev_ledger = list(incoming.get("whiteboardLedger") or incoming.get("whiteboard_ledger") or [])
    prev_turn_count = incoming.get("turnCount") or incoming.get("turn_count") or 0

    if total_agents > 0:
        director_state = {
            "turnCount": prev_turn_count + 1,
            "agentResponses": prev_responses + [{
                "agentId": current_agent_id,
                "agentName": current_agent_name or current_agent_id,
                "contentPreview": content_preview,
                "actionCount": agent_action_count,
                "whiteboardActions": list(agent_wb_actions),
            }],
            "whiteboardLedger": prev_ledger + list(agent_wb_actions),
        }
    else:
        director_state = {
            "turnCount": prev_turn_count,
            "agentResponses": prev_responses,
            "whiteboardLedger": prev_ledger,
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
        "Orchestration complete. agents=%d actions=%d hadContent=%s turnCount=%d",
        total_agents, total_actions, agent_had_content, director_state["turnCount"],
    )
