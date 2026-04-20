"""ModeMCP — controls workflow mode. Mirrors lib/pbl/mcp/mode-mcp.ts"""

from __future__ import annotations
from core.pbl.types import PBLMode, tool_ok, tool_err

_AVAILABLE: list[PBLMode] = ["project_info", "agent", "issueboard", "idle"]


class ModeMCP:
    def __init__(
        self,
        available_modes: list[PBLMode] | None = None,
        default_mode: PBLMode = "project_info",
    ) -> None:
        self._available = available_modes or _AVAILABLE
        self._current: PBLMode = default_mode

    def set_mode(self, mode: PBLMode) -> dict:
        if mode not in self._available:
            return tool_err(f'Mode "{mode}" not available. Available: {", ".join(self._available)}')
        if mode == self._current:
            return tool_err(f'Already in "{mode}" mode.')
        self._current = mode
        return tool_ok(message=f'Switched to "{mode}" mode.')

    def get_current_mode(self) -> PBLMode:
        return self._current

    def get_available_modes(self) -> list[PBLMode]:
        return list(self._available)
