"""IssueboardMCP — manages issues/workflow. Mirrors lib/pbl/mcp/issueboard-mcp.ts"""

from __future__ import annotations
from core.pbl.types import PBLProjectConfig, PBLIssue, PBLIssueboard, tool_ok, tool_err
from core.pbl.agent_mcp import AgentMCP
from core.pbl.agent_templates import get_question_agent_prompt, get_judge_agent_prompt


class IssueboardMCP:
    def __init__(
        self,
        config: PBLProjectConfig,
        agent_mcp: AgentMCP,
        language: str = "en-US",
    ) -> None:
        self._config = config
        self._agent_mcp = agent_mcp
        self._language = language
        self._next_issue_id = 1

    def create_issueboard(self) -> dict:
        self._config.issueboard = PBLIssueboard()
        self._next_issue_id = 1
        return tool_ok(message="Issueboard created successfully.")

    def get_issueboard(self) -> dict:
        return tool_ok(
            agent_ids=list(self._config.issueboard.agent_ids),
            issues=[i.to_dict() for i in self._config.issueboard.issues],
        )

    def update_issueboard_agents(self, agent_ids: list[str]) -> dict:
        self._config.issueboard.agent_ids = list(agent_ids)
        return tool_ok(message="Issueboard agents updated successfully.")

    def create_issue(
        self,
        title: str,
        description: str,
        person_in_charge: str,
        participants: list[str] | None = None,
        notes: str = "",
        parent_issue: str | None = None,
        index: int = 0,
    ) -> dict:
        if not title or not title.strip():
            return tool_err("Title cannot be empty.")
        if not person_in_charge or not person_in_charge.strip():
            return tool_err("Person in charge cannot be empty.")
        if parent_issue and not any(
            i.id == parent_issue for i in self._config.issueboard.issues
        ):
            return tool_err(f'Parent issue "{parent_issue}" not found.')

        issue_id = f"issue_{self._next_issue_id}"
        self._next_issue_id += 1
        question_agent_name = f"Question Agent - {issue_id}"
        judge_agent_name = f"Judge Agent - {issue_id}"

        new_issue = PBLIssue(
            id=issue_id,
            title=title,
            description=description,
            person_in_charge=person_in_charge,
            participants=list(participants or []),
            notes=notes,
            parent_issue=parent_issue,
            index=index,
            is_done=False,
            is_active=False,
            generated_questions="",
            question_agent_name=question_agent_name,
            judge_agent_name=judge_agent_name,
        )
        self._config.issueboard.issues.append(new_issue)

        # Auto-create question and judge agents
        self._agent_mcp.create_agent(
            name=question_agent_name,
            system_prompt=get_question_agent_prompt(self._language),
            default_mode="chat",
            actor_role="Question Assistant for Issue",
            role_division="development",
            is_system_agent=True,
        )
        self._agent_mcp.create_agent(
            name=judge_agent_name,
            system_prompt=get_judge_agent_prompt(self._language),
            default_mode="chat",
            actor_role="Judge for Issue Completion",
            role_division="management",
            is_system_agent=True,
        )

        return tool_ok(issue_id=issue_id, message="Issue created with question and judge agents.")

    def list_issues(self) -> dict:
        return tool_ok(issues=[i.to_dict() for i in self._config.issueboard.issues])

    def get_issue(self, issue_id: str) -> dict:
        issue = next((i for i in self._config.issueboard.issues if i.id == issue_id), None)
        if not issue:
            return tool_err(f'Issue "{issue_id}" not found.')
        return tool_ok(issues=[issue.to_dict()])

    def update_issue(
        self,
        issue_id: str,
        title: str | None = None,
        description: str | None = None,
        person_in_charge: str | None = None,
        participants: list[str] | None = None,
        notes: str | None = None,
        parent_issue: str | None = None,
        index: int | None = None,
    ) -> dict:
        issue = next((i for i in self._config.issueboard.issues if i.id == issue_id), None)
        if not issue:
            return tool_err(f'Issue "{issue_id}" not found.')

        if parent_issue is not None and parent_issue != "" and not any(
            i.id == parent_issue for i in self._config.issueboard.issues
        ):
            return tool_err(f'Parent issue "{parent_issue}" not found.')

        if title is not None: issue.title = title
        if description is not None: issue.description = description
        if person_in_charge is not None: issue.person_in_charge = person_in_charge
        if participants is not None: issue.participants = list(participants)
        if notes is not None: issue.notes = notes
        if parent_issue is not None: issue.parent_issue = parent_issue or None
        if index is not None: issue.index = index

        return tool_ok(message="Issue updated successfully.")

    def delete_issue(self, issue_id: str) -> dict:
        idx = next(
            (i for i, iss in enumerate(self._config.issueboard.issues) if iss.id == issue_id),
            None,
        )
        if idx is None:
            return tool_err(f'Issue "{issue_id}" not found.')
        self._config.issueboard.issues.pop(idx)
        # Remove child issues
        self._config.issueboard.issues = [
            i for i in self._config.issueboard.issues if i.parent_issue != issue_id
        ]
        return tool_ok(message="Issue deleted successfully.")

    def reorder_issues(self, issue_ids: list[str]) -> dict:
        for iid in issue_ids:
            if not any(i.id == iid for i in self._config.issueboard.issues):
                return tool_err(f'Issue "{iid}" not found.')

        reordered: list[PBLIssue] = []
        for idx, iid in enumerate(issue_ids):
            issue = next(i for i in self._config.issueboard.issues if i.id == iid)
            issue.index = idx
            reordered.append(issue)
        # Append remaining issues not in the list
        for issue in self._config.issueboard.issues:
            if issue.id not in issue_ids:
                reordered.append(issue)
        self._config.issueboard.issues = reordered
        return tool_ok(message="Issues reordered successfully.")

    def activate_next_issue(self) -> dict:
        current = next((i for i in self._config.issueboard.issues if i.is_active), None)
        if current:
            current.is_active = False
            self._config.issueboard.current_issue_id = None

        candidates = sorted(
            [i for i in self._config.issueboard.issues if not i.is_done],
            key=lambda i: i.index,
        )
        if not candidates:
            return tool_err("No more issues to activate.")

        nxt = candidates[0]
        nxt.is_active = True
        self._config.issueboard.current_issue_id = nxt.id
        return tool_ok(issue_id=nxt.id, message=f"Activated issue: {nxt.title}")

    def complete_current_issue(self) -> dict:
        current = next((i for i in self._config.issueboard.issues if i.is_active), None)
        if not current:
            return tool_err("No active issue to complete.")
        current.is_done = True
        current.is_active = False
        self._config.issueboard.current_issue_id = None
        return tool_ok(message=f'Issue "{current.id}" marked as complete.')
