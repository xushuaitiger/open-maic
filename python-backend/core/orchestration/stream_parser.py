"""
Incremental structured JSON array parser for streaming LLM output.

The LLM is expected to produce:
  [{"type":"action","name":"spotlight","params":{"elementId":"img_1"}},
   {"type":"text","content":"Hello students..."},
   ...]

This parser accumulates chunks and emits new complete items as they appear,
plus partial text deltas for the trailing incomplete text item.

Mirrors lib/orchestration/stateless-generate.ts (parseStructuredChunk, finalizeParser).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedAction:
    action_id: str
    action_name: str
    params: dict[str, Any]


@dataclass
class OrderedEntry:
    type: str   # "text" | "action"
    index: int


@dataclass
class ParseResult:
    text_chunks: list[str] = field(default_factory=list)
    actions: list[ParsedAction] = field(default_factory=list)
    ordered: list[OrderedEntry] = field(default_factory=list)
    is_done: bool = False


@dataclass
class ParserState:
    buffer: str = ""
    json_started: bool = False
    last_parsed_item_count: int = 0
    last_partial_text_length: int = 0
    is_done: bool = False


def create_parser_state() -> ParserState:
    return ParserState()


try:  # optional but strongly recommended
    from json_repair import repair_json as _repair_json
except ImportError:  # graceful fallback when dep is not installed yet
    _repair_json = None  # type: ignore[assignment]


def _try_parse_buffer(buffer: str) -> list | None:
    """Try to parse the buffer as a (potentially incomplete) JSON array.

    Strategy (in order):
      1. Strict json.loads
      2. Close the array with ']'/'}]' and retry
      3. json-repair (handles trailing commas, unbalanced braces, partial strings)
      4. Brute-force scan for top-level complete `{...}` objects inside the array
    """
    try:
        parsed = json.loads(buffer)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    for suffix in ("]", "}]", '"}]', '"}}]'):
        try:
            parsed = json.loads(buffer.rstrip() + suffix)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, Exception):
            pass

    if _repair_json is not None:
        try:
            repaired = _repair_json(buffer, return_objects=True)
            if isinstance(repaired, list):
                return repaired
            if isinstance(repaired, dict):
                return [repaired]
        except Exception:
            pass

    results: list = []
    depth = 0
    in_string = False
    escaped = False
    start = -1
    i = buffer.find("[")
    if i == -1:
        return None
    i += 1
    while i < len(buffer):
        ch = buffer[i]
        if escaped:
            escaped = False
        elif ch == "\\" and in_string:
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(buffer[start:i + 1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        i += 1
    return results if results else None


def parse_chunk(chunk: str, state: ParserState) -> ParseResult:
    """Parse a new streaming chunk and return newly emitted items."""
    result = ParseResult()

    if state.is_done:
        return result

    state.buffer += chunk

    # Find opening `[`
    if not state.json_started:
        idx = state.buffer.find("[")
        if idx == -1:
            return result
        state.buffer = state.buffer[idx:]
        state.json_started = True

    # Check if array is closed
    trimmed = state.buffer.rstrip()
    is_array_closed = trimmed.endswith("]") and len(trimmed) > 1

    parsed = _try_parse_buffer(state.buffer)
    if not parsed:
        return result

    complete_up_to = len(parsed) if is_array_closed else max(0, len(parsed) - 1)

    # Emit newly completed items
    for i in range(state.last_parsed_item_count, complete_up_to):
        item = parsed[i]
        if not isinstance(item, dict):
            continue

        # If this was the trailing partial text item, only emit the delta
        if (
            i == state.last_parsed_item_count
            and state.last_partial_text_length > 0
            and item.get("type") == "text"
        ):
            content = item.get("content", "")
            remaining = content[state.last_partial_text_length:]
            if remaining:
                result.text_chunks.append(remaining)
                result.ordered.append(OrderedEntry("text", len(result.text_chunks) - 1))
            state.last_partial_text_length = 0
            continue

        _emit_item(item, result)

    state.last_parsed_item_count = complete_up_to

    # Stream partial text delta for trailing incomplete item
    if not is_array_closed and len(parsed) > complete_up_to:
        last_item = parsed[-1]
        if isinstance(last_item, dict) and last_item.get("type") == "text":
            content = last_item.get("content", "")
            if len(content) > state.last_partial_text_length:
                delta = content[state.last_partial_text_length:]
                result.text_chunks.append(delta)
                result.ordered.append(OrderedEntry("text", len(result.text_chunks) - 1))
                state.last_partial_text_length = len(content)

    if is_array_closed:
        state.is_done = True
        result.is_done = True
        state.last_parsed_item_count = len(parsed)
        state.last_partial_text_length = 0

    return result


def finalize_parser(state: ParserState) -> ParseResult:
    """Finalize parsing after stream ends — handles plain-text fallback."""
    result = ParseResult(is_done=True)

    if state.is_done:
        return result

    content = state.buffer.strip()
    if not content:
        return result

    if not state.json_started:
        result.text_chunks.append(content)
        result.ordered.append(OrderedEntry("text", 0))
    else:
        final = parse_chunk("", state)
        result.text_chunks.extend(final.text_chunks)
        result.actions.extend(final.actions)
        result.ordered.extend(final.ordered)

        if not result.text_chunks and not result.actions:
            bracket = content.find("[")
            raw = content[bracket + 1:].strip() if bracket != -1 else content
            if raw:
                result.text_chunks.append(raw)
                result.ordered.append(OrderedEntry("text", 0))

    state.is_done = True
    return result


def _emit_item(item: dict, result: ParseResult) -> None:
    if item.get("type") == "text":
        content = item.get("content", "")
        if content:
            result.text_chunks.append(content)
            result.ordered.append(OrderedEntry("text", len(result.text_chunks) - 1))
    elif item.get("type") == "action":
        import time, random
        action = ParsedAction(
            action_id=item.get("action_id", f"action-{int(time.time() * 1000)}-{random.randint(0, 9999)}"),
            action_name=str(item.get("name") or item.get("tool_name", "")),
            params=dict(item.get("params") or item.get("parameters") or {}),
        )
        result.actions.append(action)
        result.ordered.append(OrderedEntry("action", len(result.actions) - 1))
