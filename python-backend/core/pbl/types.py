"""
PBL (Project-Based Learning) type definitions.
Mirrors lib/pbl/types.ts
"""

from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass, field


PBLMode = Literal["project_info", "agent", "issueboard", "idle"]
PBLRoleDivision = Literal["management", "development"]


@dataclass
class PBLProjectInfo:
    title: str = ""
    description: str = ""


@dataclass
class PBLAgent:
    name: str
    actor_role: str = ""
    role_division: PBLRoleDivision = "development"
    system_prompt: str = ""
    default_mode: str = "chat"
    delay_time: int = 0
    env: dict[str, Any] = field(default_factory=dict)
    is_user_role: bool = False
    is_active: bool = False
    is_system_agent: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "actor_role": self.actor_role,
            "role_division": self.role_division,
            "system_prompt": self.system_prompt,
            "default_mode": self.default_mode,
            "delay_time": self.delay_time,
            "env": self.env,
            "is_user_role": self.is_user_role,
            "is_active": self.is_active,
            "is_system_agent": self.is_system_agent,
        }


@dataclass
class PBLIssue:
    id: str
    title: str
    description: str
    person_in_charge: str
    participants: list[str] = field(default_factory=list)
    notes: str = ""
    parent_issue: str | None = None
    index: int = 0
    is_done: bool = False
    is_active: bool = False
    generated_questions: str = ""
    question_agent_name: str = ""
    judge_agent_name: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "person_in_charge": self.person_in_charge,
            "participants": self.participants,
            "notes": self.notes,
            "parent_issue": self.parent_issue,
            "index": self.index,
            "is_done": self.is_done,
            "is_active": self.is_active,
            "generated_questions": self.generated_questions,
            "question_agent_name": self.question_agent_name,
            "judge_agent_name": self.judge_agent_name,
        }


@dataclass
class PBLIssueboard:
    agent_ids: list[str] = field(default_factory=list)
    issues: list[PBLIssue] = field(default_factory=list)
    current_issue_id: str | None = None


@dataclass
class PBLChatMessage:
    id: str
    agent_name: str
    message: str
    timestamp: int
    read_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "message": self.message,
            "timestamp": self.timestamp,
            "read_by": self.read_by,
        }


@dataclass
class PBLChat:
    messages: list[PBLChatMessage] = field(default_factory=list)


@dataclass
class PBLProjectConfig:
    project_info: PBLProjectInfo = field(default_factory=PBLProjectInfo)
    agents: list[PBLAgent] = field(default_factory=list)
    issueboard: PBLIssueboard = field(default_factory=PBLIssueboard)
    chat: PBLChat = field(default_factory=PBLChat)
    selected_role: str | None = None

    def to_dict(self) -> dict:
        return {
            "projectInfo": {
                "title": self.project_info.title,
                "description": self.project_info.description,
            },
            "agents": [a.to_dict() for a in self.agents],
            "issueboard": {
                "agent_ids": self.issueboard.agent_ids,
                "issues": [i.to_dict() for i in self.issueboard.issues],
                "current_issue_id": self.issueboard.current_issue_id,
            },
            "chat": {
                "messages": [m.to_dict() for m in self.chat.messages],
            },
            "selectedRole": self.selected_role,
        }


# Shared tool result type
def tool_ok(**kwargs) -> dict:
    return {"success": True, **kwargs}


def tool_err(error: str) -> dict:
    return {"success": False, "error": error}
