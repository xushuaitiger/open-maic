"""
Single entry point for all prompt lookups in the generation pipeline.

There are currently TWO prompt sources in this codebase:

  1. ``core/generation/prompts/templates/<id>/system.md + user.md``
     Editable Markdown templates with snippet includes — preferred for any
     new prompt because operators can iterate on copy without redeploying.

  2. ``core/generation/prompt_strings.py``
     Hardcoded Python strings — legacy.  Kept because it currently drives
     ``classroom_generator.py`` and rewriting the call-sites to match the
     template variables is a separate refactor.

This module exposes a uniform ``build_prompt(prompt_id, variables)``:

  • If a template exists at ``templates/<prompt_id>/system.md`` it is used
    (variables are interpolated).
  • Otherwise the call returns ``None`` and the caller is expected to fall
    back to the legacy ``prompt_strings`` constants.

Use this everywhere instead of importing ``loader`` or ``prompt_strings``
directly — that way the cut-over to file-based-only is a one-file change.
"""

from __future__ import annotations

from typing import Any

from core.generation.prompts.loader import build_prompt as _build_template

__all__ = ["build_prompt", "list_template_ids"]


def build_prompt(prompt_id: str, variables: dict[str, Any]) -> dict[str, str] | None:
    """Render a prompt by id. Returns ``{"system","user"}`` or None if missing."""
    return _build_template(prompt_id, variables)


def list_template_ids() -> list[str]:
    from pathlib import Path
    root = Path(__file__).parent / "prompts" / "templates"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
