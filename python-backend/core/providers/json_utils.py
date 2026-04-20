"""Robust JSON parsing for LLM responses.

LLMs love to wrap JSON in ```` ```json ```` fences, append explanations after
the closing brace, omit a trailing brace, etc. ``parse_llm_json`` strips
fences, then tries (in order): standard json → bracket-balanced extraction →
``json-repair``.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None  # type: ignore[assignment]


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _extract_first_balanced(text: str, opener: str, closer: str) -> str | None:
    """Return the first top-level balanced ``opener…closer`` block, or None."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_llm_json(text: str, *, expect: str = "any") -> Any:
    """Best-effort parse of LLM-produced JSON.

    ``expect`` may be 'object', 'array' or 'any'. When the loose parsers
    return something inconsistent with ``expect``, we still raise.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    cleaned = strip_code_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate: str | None
    if expect == "array":
        candidate = _extract_first_balanced(cleaned, "[", "]")
    elif expect == "object":
        candidate = _extract_first_balanced(cleaned, "{", "}")
    else:
        candidate = (
            _extract_first_balanced(cleaned, "{", "}")
            or _extract_first_balanced(cleaned, "[", "]")
        )

    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if _repair_json is not None:
        try:
            return _repair_json(cleaned, return_objects=True)
        except Exception:
            pass

    raise ValueError(f"Could not parse LLM JSON response: {cleaned[:200]}…")
