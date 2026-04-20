"""
Prompt loader — reads .md templates from disk and performs variable interpolation.

Mirrors lib/generation/prompts/loader.ts exactly:
  - Templates live in prompts/templates/{promptId}/system.md + user.md
  - Snippets are in prompts/snippets/{snippetId}.md
  - {{snippet:name}} in templates is replaced with snippet content
  - {{variable}} is replaced with values from a variables dict
  - Results are cached in-process
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("prompt_loader")

# Root of the prompts directory (same package as this file)
_PROMPTS_DIR = Path(__file__).parent

_prompt_cache: dict[str, "LoadedPrompt"] = {}
_snippet_cache: dict[str, str] = {}


@dataclass
class LoadedPrompt:
    id: str
    system_prompt: str
    user_prompt_template: str


# ---------------------------------------------------------------------------
# Snippet loading
# ---------------------------------------------------------------------------

def load_snippet(snippet_id: str) -> str:
    if snippet_id in _snippet_cache:
        return _snippet_cache[snippet_id]

    path = _PROMPTS_DIR / "snippets" / f"{snippet_id}.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
        _snippet_cache[snippet_id] = content
        return content
    except FileNotFoundError:
        log.warning("Snippet not found: %s", snippet_id)
        return f"{{{{snippet:{snippet_id}}}}}"


def _process_snippets(template: str) -> str:
    """Replace {{snippet:name}} with actual snippet content."""
    return re.sub(
        r"\{\{snippet:([\w-]+)\}\}",
        lambda m: load_snippet(m.group(1)),
        template,
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(prompt_id: str) -> LoadedPrompt | None:
    if prompt_id in _prompt_cache:
        return _prompt_cache[prompt_id]

    prompt_dir = _PROMPTS_DIR / "templates" / prompt_id

    try:
        system_path = prompt_dir / "system.md"
        system_prompt = _process_snippets(system_path.read_text(encoding="utf-8").strip())

        user_prompt_template = ""
        user_path = prompt_dir / "user.md"
        if user_path.exists():
            user_prompt_template = _process_snippets(user_path.read_text(encoding="utf-8").strip())

        loaded = LoadedPrompt(
            id=prompt_id,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        _prompt_cache[prompt_id] = loaded
        return loaded

    except Exception as exc:
        log.error("Failed to load prompt %s: %s", prompt_id, exc)
        return None


# ---------------------------------------------------------------------------
# Variable interpolation
# ---------------------------------------------------------------------------

def interpolate_variables(template: str, variables: dict[str, Any]) -> str:
    """Replace {{variable}} with values from the variables dict."""
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        value = variables.get(key)
        if value is None:
            return match.group(0)
        if isinstance(value, (dict, list)):
            import json
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    return re.sub(r"\{\{(\w+)\}\}", replacer, template)


# ---------------------------------------------------------------------------
# High-level build
# ---------------------------------------------------------------------------

def build_prompt(
    prompt_id: str,
    variables: dict[str, Any],
) -> dict[str, str] | None:
    """Load a prompt and interpolate variables. Returns {system, user} or None."""
    prompt = load_prompt(prompt_id)
    if not prompt:
        return None
    return {
        "system": interpolate_variables(prompt.system_prompt, variables),
        "user": interpolate_variables(prompt.user_prompt_template, variables),
    }


def clear_cache() -> None:
    _prompt_cache.clear()
    _snippet_cache.clear()
