"""AgentMCP — manages project agent roles. Mirrors lib/pbl/mcp/agent-mcp.ts"""

from __future__ import annotations
from core.pbl.types import PBLProjectConfig, PBLAgent, PBLRoleDivision, tool_ok, tool_err


class AgentMCP:
    def __init__(self, config: PBLProjectConfig) -> None:
        self._config = config

    def list_agents(self) -> dict:
        agents = [a.to_dict() for a in self._config.agents]
        return tool_ok(
            agents=agents,
            message="No agents found." if not agents else None,
        )

    def get_agent_info(self, name: str) -> dict:
        agent = next((a for a in self._config.agents if a.name == name), None)
        if not agent:
            return tool_err(f'Agent "{name}" not found.')
        return tool_ok(agent=agent.to_dict())

    def create_agent(
        self,
        name: str,
        system_prompt: str,
        default_mode: str = "chat",
        delay_time: int = 0,
        actor_role: str = "",
        role_division: PBLRoleDivision = "development",
        is_system_agent: bool = False,
    ) -> dict:
        if not name or not name.strip():
            return tool_err("Agent name cannot be empty.")
        if not system_prompt or not system_prompt.strip():
            return tool_err("System prompt cannot be empty.")
        if any(a.name == name for a in self._config.agents):
            return tool_err(f'Agent "{name}" already exists.')

        new_agent = PBLAgent(
            name=name,
            actor_role=actor_role,
            role_division=role_division,
            system_prompt=system_prompt,
            default_mode=default_mode,
            delay_time=delay_time,
            env={"chat": {"max_tokens": 4096, "system_prompt": system_prompt}},
            is_user_role=False,
            is_active=False,
            is_system_agent=is_system_agent,
        )
        self._config.agents.append(new_agent)
        return tool_ok(message=f'Agent "{name}" created successfully.')

    def update_agent(
        self,
        name: str,
        new_name: str | None = None,
        system_prompt: str | None = None,
        default_mode: str | None = None,
        delay_time: int | None = None,
        actor_role: str | None = None,
        role_division: PBLRoleDivision | None = None,
    ) -> dict:
        agent = next((a for a in self._config.agents if a.name == name), None)
        if not agent:
            return tool_err(f'Agent "{name}" not found.')

        if new_name and new_name != name and any(a.name == new_name for a in self._config.agents):
            return tool_err(f'Agent "{new_name}" already exists.')

        if new_name is not None:
            agent.name = new_name
        if system_prompt is not None:
            agent.system_prompt = system_prompt
            if isinstance(agent.env.get("chat"), dict):
                agent.env["chat"]["system_prompt"] = system_prompt
        if default_mode is not None:
            agent.default_mode = default_mode
        if delay_time is not None:
            agent.delay_time = delay_time
        if actor_role is not None:
            agent.actor_role = actor_role
        if role_division is not None:
            agent.role_division = role_division

        return tool_ok(message="Agent updated successfully.")

    def delete_agent(self, name: str) -> dict:
        idx = next((i for i, a in enumerate(self._config.agents) if a.name == name), None)
        if idx is None:
            return tool_err(f'Agent "{name}" not found.')
        self._config.agents.pop(idx)
        return tool_ok(message="Agent deleted successfully.")
