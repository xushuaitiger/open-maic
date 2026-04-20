"""ProjectMCP — manages project info. Mirrors lib/pbl/mcp/project-mcp.ts"""

from __future__ import annotations
from core.pbl.types import PBLProjectConfig, tool_ok, tool_err


class ProjectMCP:
    def __init__(self, config: PBLProjectConfig) -> None:
        self._config = config

    def get_project_info(self) -> dict:
        return tool_ok(
            title=self._config.project_info.title,
            description=self._config.project_info.description,
        )

    def update_title(self, title: str) -> dict:
        if not title or not title.strip():
            return tool_err("Title cannot be empty.")
        self._config.project_info.title = title
        return tool_ok(message="Title updated successfully.")

    def update_description(self, description: str) -> dict:
        if description is None:
            return tool_err("Description cannot be null.")
        self._config.project_info.description = description
        return tool_ok(message="Description updated successfully.")
