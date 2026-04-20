"""
PBL Generation — agentic tool-calling loop.

Mirrors lib/pbl/generate-pbl.ts:
  Uses the LLM's native tool-use / function-calling to drive a multi-step loop
  where the model designs the full PBL project by calling MCP tools.

  Steps:
  1. project_info mode  → set title + description
  2. agent mode         → create 2-4 student roles
  3. issueboard mode    → create N sequential issues
  4. idle mode          → done

  After the loop, post-processing activates the first issue and generates
  initial guiding questions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from core.pbl.types import PBLProjectConfig, PBLChatMessage
from core.pbl.mode_mcp import ModeMCP
from core.pbl.project_mcp import ProjectMCP
from core.pbl.agent_mcp import AgentMCP
from core.pbl.issueboard_mcp import IssueboardMCP
from core.pbl.system_prompt import PBLSystemPromptConfig, build_pbl_system_prompt
from core.providers.llm import ResolvedModel, call_llm_with_tools

log = logging.getLogger("generate_pbl")

MAX_STEPS = 30


@dataclass
class GeneratePBLConfig:
    project_topic: str
    project_description: str
    target_skills: list[str]
    issue_count: int = 3
    language: str = "en-US"


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI function-call format)
# ---------------------------------------------------------------------------

def _build_tools(mode_mcp: ModeMCP) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "set_mode",
                "description": "Switch the current working mode. Available modes: project_info, agent, issueboard, idle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["project_info", "agent", "issueboard", "idle"]},
                    },
                    "required": ["mode"],
                },
            },
        },
        # project_info tools
        {
            "type": "function",
            "function": {
                "name": "get_project_info",
                "description": "Get the current project information. Requires project_info mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_title",
                "description": "Update the project title. Requires project_info mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_description",
                "description": "Update the project description. Requires project_info mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            },
        },
        # agent tools
        {
            "type": "function",
            "function": {
                "name": "list_project_agents",
                "description": "List all agent roles. Requires agent mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_agent",
                "description": "Create a new agent role. Requires agent mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "system_prompt": {"type": "string"},
                        "default_mode": {"type": "string"},
                        "actor_role": {"type": "string"},
                        "role_division": {"type": "string", "enum": ["management", "development"]},
                    },
                    "required": ["name", "system_prompt", "default_mode"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_agent",
                "description": "Update an agent role. Requires agent mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "new_name": {"type": "string"},
                        "system_prompt": {"type": "string"},
                        "default_mode": {"type": "string"},
                        "actor_role": {"type": "string"},
                        "role_division": {"type": "string", "enum": ["management", "development"]},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_agent",
                "description": "Delete an agent role. Requires agent mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        # issueboard tools
        {
            "type": "function",
            "function": {
                "name": "create_issueboard",
                "description": "Create/reset the issueboard. Requires issueboard mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_issueboard",
                "description": "Get the issueboard. Requires issueboard mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_issueboard_agents",
                "description": "Update issueboard agent list. Requires issueboard mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"agent_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["agent_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_issue",
                "description": "Create an issue. Auto-creates Question and Judge agents. Requires issueboard mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "person_in_charge": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                        "parent_issue": {"type": "string"},
                        "index": {"type": "integer"},
                    },
                    "required": ["title", "description", "person_in_charge"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_issues",
                "description": "List all issues. Requires issueboard mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_issue",
                "description": "Update an issue. Requires issueboard mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "person_in_charge": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                        "parent_issue": {"type": "string"},
                        "index": {"type": "integer"},
                    },
                    "required": ["issue_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_issue",
                "description": "Delete an issue. Requires issueboard mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"issue_id": {"type": "string"}},
                    "required": ["issue_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reorder_issues",
                "description": "Reorder issues. Requires issueboard mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"issue_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["issue_ids"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch(
    tool_name: str,
    args: dict,
    mode_mcp: ModeMCP,
    project_mcp: ProjectMCP,
    agent_mcp: AgentMCP,
    issueboard_mcp: IssueboardMCP,
) -> dict:
    mode = mode_mcp.get_current_mode()

    if tool_name == "set_mode":
        return mode_mcp.set_mode(args["mode"])

    # project_info tools
    if tool_name == "get_project_info":
        return project_mcp.get_project_info() if mode == "project_info" else {"success": False, "error": "Must be in project_info mode."}
    if tool_name == "update_title":
        return project_mcp.update_title(args["title"]) if mode == "project_info" else {"success": False, "error": "Must be in project_info mode."}
    if tool_name == "update_description":
        return project_mcp.update_description(args["description"]) if mode == "project_info" else {"success": False, "error": "Must be in project_info mode."}

    # agent tools
    if tool_name == "list_project_agents":
        return agent_mcp.list_agents() if mode == "agent" else {"success": False, "error": "Must be in agent mode."}
    if tool_name == "create_agent":
        return agent_mcp.create_agent(**args) if mode == "agent" else {"success": False, "error": "Must be in agent mode."}
    if tool_name == "update_agent":
        return agent_mcp.update_agent(**args) if mode == "agent" else {"success": False, "error": "Must be in agent mode."}
    if tool_name == "delete_agent":
        return agent_mcp.delete_agent(args["name"]) if mode == "agent" else {"success": False, "error": "Must be in agent mode."}

    # issueboard tools
    if tool_name == "create_issueboard":
        return issueboard_mcp.create_issueboard() if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "get_issueboard":
        return issueboard_mcp.get_issueboard() if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "update_issueboard_agents":
        return issueboard_mcp.update_issueboard_agents(args["agent_ids"]) if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "create_issue":
        return issueboard_mcp.create_issue(**args) if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "list_issues":
        return issueboard_mcp.list_issues() if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "update_issue":
        return issueboard_mcp.update_issue(**args) if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "delete_issue":
        return issueboard_mcp.delete_issue(args["issue_id"]) if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}
    if tool_name == "reorder_issues":
        return issueboard_mcp.reorder_issues(args["issue_ids"]) if mode == "issueboard" else {"success": False, "error": "Must be in issueboard mode."}

    return {"success": False, "error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

async def generate_pbl_content(
    config: GeneratePBLConfig,
    resolved_model: ResolvedModel,
    on_progress: Callable[[str], None] | None = None,
) -> PBLProjectConfig:
    """
    Run the agentic PBL generation loop.
    Returns a fully populated PBLProjectConfig.
    """
    project_config = PBLProjectConfig()

    mode_mcp = ModeMCP()
    project_mcp = ProjectMCP(project_config)
    agent_mcp = AgentMCP(project_config)
    issueboard_mcp = IssueboardMCP(project_config, agent_mcp, config.language)

    tools = _build_tools(mode_mcp)
    system_prompt = build_pbl_system_prompt(
        PBLSystemPromptConfig(
            project_topic=config.project_topic,
            project_description=config.project_description,
            target_skills=config.target_skills,
            issue_count=config.issue_count,
            language=config.language,
        )
    )

    user_msg = (
        "请设计一个PBL项目。现在从 project_info 模式开始，先设置项目标题和描述。"
        if config.language == "zh-CN"
        else "Design a PBL project. Start in project_info mode by setting the project title and description."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    on_progress and on_progress("Starting PBL project generation...")

    for step in range(MAX_STEPS):
        # Check if idle (done)
        if mode_mcp.get_current_mode() == "idle":
            on_progress and on_progress("Mode reached idle. Generation complete.")
            break

        result = await call_llm_with_tools(resolved_model, messages, tools)

        # Append assistant message
        messages.append({"role": "assistant", "content": result.get("content") or "", "tool_calls": result.get("tool_calls") or []})

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            # No more tool calls — LLM finished
            on_progress and on_progress(f"Step {step}: no tool calls, stopping.")
            break

        # Execute each tool call and append results
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            on_progress and on_progress(f"Tool: {fn_name}")
            tool_result = _dispatch(fn_name, fn_args, mode_mcp, project_mcp, agent_mcp, issueboard_mcp)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

            # Stop immediately when idle is set
            if fn_name == "set_mode" and fn_args.get("mode") == "idle" and tool_result.get("success"):
                break

    if mode_mcp.get_current_mode() != "idle":
        on_progress and on_progress(
            "Warning: generation did not reach idle mode. Project may be incomplete."
        )

    # ── Invariants: verify the LLM produced a usable project ────────────────
    violations = _check_invariants(project_config, config)
    if violations:
        msg = "PBL invariants violated: " + "; ".join(violations)
        on_progress and on_progress(msg)
        log.warning(msg)
        # Auto-repair the cheap violations rather than failing outright.
        _repair_invariants(project_config, project_mcp, config)
        remaining = _check_invariants(project_config, config)
        if remaining:
            raise PBLGenerationError(
                "PBL generation produced an incomplete project: "
                + "; ".join(remaining)
            )

    on_progress and on_progress("PBL structure generated. Running post-processing...")
    await _post_process(project_config, resolved_model, config.language, on_progress)
    on_progress and on_progress("PBL project generation complete!")

    return project_config


# ── Invariants ──────────────────────────────────────────────────────────────

class PBLGenerationError(RuntimeError):
    """Raised when the agentic loop produces a structurally invalid project."""


def _check_invariants(
    project: PBLProjectConfig,
    cfg: GeneratePBLConfig,
) -> list[str]:
    """Return a list of human-readable invariant violations (empty = healthy)."""
    issues: list[str] = []

    if not (project.project_info.title or "").strip():
        issues.append("project title is empty")
    if not (project.project_info.description or "").strip():
        issues.append("project description is empty")

    agent_count = len(project.agents)
    if agent_count < 2:
        issues.append(f"need at least 2 agents, got {agent_count}")

    issue_count = len(project.issueboard.issues)
    if issue_count < 1:
        issues.append(f"need at least 1 issue, got {issue_count}")
    elif cfg.issue_count and issue_count < max(1, cfg.issue_count // 2):
        # Allow some slack but flag a serious shortfall.
        issues.append(
            f"requested {cfg.issue_count} issues but only got {issue_count}"
        )

    # Each issue's person_in_charge must reference a known agent.
    agent_names = {a.name for a in project.agents}
    for it in project.issueboard.issues:
        if it.person_in_charge and it.person_in_charge not in agent_names:
            issues.append(
                f"issue '{it.title}' assigned to unknown agent '{it.person_in_charge}'"
            )

    return issues


def _repair_invariants(
    project: PBLProjectConfig,
    project_mcp: ProjectMCP,
    cfg: GeneratePBLConfig,
) -> None:
    """Best-effort repair: fill in defaults so the project is at least usable."""
    if not (project.project_info.title or "").strip():
        project_mcp.update_title(cfg.project_topic[:100] or "Untitled PBL Project")
    if not (project.project_info.description or "").strip():
        project_mcp.update_description(
            (cfg.project_description or cfg.project_topic)[:500]
        )

    # If the LLM forgot to assign issues to agents, fall back to the first agent.
    if project.agents:
        fallback = project.agents[0].name
        for it in project.issueboard.issues:
            if not it.person_in_charge or it.person_in_charge not in {a.name for a in project.agents}:
                it.person_in_charge = fallback


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

async def _post_process(
    config: PBLProjectConfig,
    resolved_model: ResolvedModel,
    language: str,
    on_progress: Callable[[str], None] | None,
) -> None:
    """Activate first issue, generate initial questions, add welcome chat message."""
    if not config.issueboard.issues:
        return

    sorted_issues = sorted(config.issueboard.issues, key=lambda i: i.index)
    first_issue = sorted_issues[0]
    first_issue.is_active = True
    config.issueboard.current_issue_id = first_issue.id

    on_progress and on_progress(f"Activating first issue: {first_issue.title}")

    question_agent = next(
        (a for a in config.agents if a.name == first_issue.question_agent_name), None
    )
    if not question_agent:
        on_progress and on_progress("Warning: Question agent not found for first issue.")
        return

    try:
        on_progress and on_progress("Generating initial questions for first issue...")

        if language == "zh-CN":
            context = f"""## 任务信息

**标题**: {first_issue.title}
**描述**: {first_issue.description}
**负责人**: {first_issue.person_in_charge}
{f"**参与者**: {'、'.join(first_issue.participants)}" if first_issue.participants else ""}
{f"**备注**: {first_issue.notes}" if first_issue.notes else ""}

## 你的任务

根据以上任务信息，生成1-3个具体、可操作的引导问题，帮助学生理解和完成这个任务。请以编号列表格式回答。"""
            welcome_prefix = f'你好！我是这个任务的提问助手："{first_issue.title}"\n\n为了引导你的学习，我准备了一些问题：\n\n'
            welcome_suffix = "\n\n随时 @question 我来获取帮助或澄清！"
        else:
            context = f"""## Issue Information

**Title**: {first_issue.title}
**Description**: {first_issue.description}
**Person in Charge**: {first_issue.person_in_charge}
{f"**Participants**: {', '.join(first_issue.participants)}" if first_issue.participants else ""}
{f"**Notes**: {first_issue.notes}" if first_issue.notes else ""}

## Your Task

Based on the issue information above, generate 1-3 specific, actionable questions to guide students. Format your response as a numbered list."""
            welcome_prefix = f'Hello! I\'m your Question Agent for this issue: "{first_issue.title}"\n\nTo help guide your work, I\'ve prepared some questions:\n\n'
            welcome_suffix = "\n\nFeel free to @question me anytime if you need help or clarification!"

        questions_text = await call_llm_with_tools(
            resolved_model,
            [
                {"role": "system", "content": question_agent.system_prompt},
                {"role": "user", "content": context},
            ],
            tools=None,
            text_only=True,
        )

        first_issue.generated_questions = questions_text

        config.chat.messages.append(
            PBLChatMessage(
                id=f"msg_welcome_{int(time.time() * 1000)}",
                agent_name=first_issue.question_agent_name,
                message=welcome_prefix + questions_text + welcome_suffix,
                timestamp=int(time.time() * 1000),
            )
        )
        on_progress and on_progress("Initial questions generated and welcome message added.")

    except Exception as exc:
        on_progress and on_progress(f"Warning: Failed to generate initial questions: {exc}")
