"""
LLM provider factory.

Supports:
  - openai          → OpenAI Python SDK
  - anthropic       → Anthropic Python SDK
  - google          → google-generativeai SDK
  - All OpenAI-compatible providers (deepseek, qwen, kimi, glm, siliconflow, doubao, minimax, ...)
    via openai.AsyncOpenAI with custom base_url

Model string format: "provider_id:model_id"
  e.g.  "openai:gpt-4o-mini"
        "anthropic:claude-3-5-haiku-20241022"
        "google:gemini-2.0-flash"
        "deepseek:deepseek-chat"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from core.providers.retry import provider_semaphore, with_retry

log = logging.getLogger("llm")

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

_OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "minimax": "https://api.minimax.chat/v1",
}

_ANTHROPIC_PROVIDERS = {"anthropic", "minimax-anthropic"}
_GOOGLE_PROVIDERS = {"google"}
_OPENAI_NATIVE = {"openai"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResolvedModel:
    provider_id: str
    model_id: str
    api_key: str
    base_url: str
    client_type: str  # "openai" | "anthropic" | "google"


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse 'provider:model' → (provider_id, model_id)."""
    if ":" in model_string:
        provider_id, model_id = model_string.split(":", 1)
        return provider_id.strip(), model_id.strip()
    # Default: treat entire string as model id under openai
    return "openai", model_string.strip()


def resolve_model(
    model_string: str,
    api_key: str = "",
    base_url: str = "",
    server_config=None,
) -> ResolvedModel:
    """
    Resolve a model string into a ResolvedModel with credentials.
    server_config is a ServerConfig instance (optional).
    """
    provider_id, model_id = parse_model_string(model_string)

    if server_config:
        resolved_key = server_config.resolve_api_key(provider_id, api_key)
        resolved_url = server_config.resolve_base_url(provider_id, base_url)
    else:
        resolved_key = api_key
        resolved_url = base_url

    # Determine default base URL for OpenAI-compatible providers
    if not resolved_url and provider_id in _OPENAI_COMPATIBLE_BASE_URLS:
        resolved_url = _OPENAI_COMPATIBLE_BASE_URLS[provider_id]

    # Determine client type
    if provider_id in _ANTHROPIC_PROVIDERS:
        client_type = "anthropic"
    elif provider_id in _GOOGLE_PROVIDERS:
        client_type = "google"
    else:
        client_type = "openai"

    return ResolvedModel(
        provider_id=provider_id,
        model_id=model_id,
        api_key=resolved_key,
        base_url=resolved_url,
        client_type=client_type,
    )


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def call_llm(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
) -> str:
    """Call an LLM and return the full text response."""
    if resolved.client_type == "anthropic":
        return await _call_anthropic(resolved, messages, max_tokens, temperature)
    elif resolved.client_type == "google":
        return await _call_google(resolved, messages, max_tokens, temperature)
    else:
        return await _call_openai_compatible(resolved, messages, max_tokens, temperature)


async def stream_llm(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float = 0.7,
):
    """Yield text chunks from an LLM stream."""
    if resolved.client_type == "anthropic":
        async for chunk in _stream_anthropic(resolved, messages, max_tokens, temperature):
            yield chunk
    elif resolved.client_type == "google":
        async for chunk in _stream_google(resolved, messages, max_tokens, temperature):
            yield chunk
    else:
        async for chunk in _stream_openai_compatible(resolved, messages, max_tokens, temperature):
            yield chunk


async def call_llm_with_tools(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    text_only: bool = False,
) -> Any:
    """
    Call an LLM with optional tool/function definitions.

    Tools are described in the OpenAI ChatCompletion format::

        {"type": "function",
         "function": {"name": "...", "description": "...", "parameters": {...JSON Schema...}}}

    Returns:
      • If ``text_only=True``: the plain text response.
      • Otherwise a dict ``{"content": str, "tool_calls": [...]}`` where each
        tool_call is ``{"id": str, "function": {"name": str, "arguments": str}}``
        — ``arguments`` is a JSON-encoded string, identical across providers.

    All three native SDKs (OpenAI, Anthropic, Google) are supported.  When
    tools is empty/None we fall back to ``call_llm`` (cheaper code path).
    """
    if text_only or not tools:
        text = await call_llm(resolved, messages, max_tokens, temperature)
        return text if text_only else {"content": text, "tool_calls": []}

    if resolved.client_type == "anthropic":
        return await _call_anthropic_with_tools(resolved, messages, tools, max_tokens, temperature)
    if resolved.client_type == "google":
        return await _call_google_with_tools(resolved, messages, tools, max_tokens, temperature)
    return await _call_openai_with_tools(resolved, messages, tools, max_tokens, temperature)


# ── OpenAI tool calling ────────────────────────────────────────────────────

async def _call_openai_with_tools(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> dict:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {}
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key

    client = AsyncOpenAI(**kwargs)

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": messages,
        "temperature": temperature,
        "tools": tools,
        "tool_choice": "auto",
    }
    if max_tokens:
        params["max_tokens"] = max_tokens

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: client.chat.completions.create(**params),
            label=f"{resolved.provider_id}.chat.tools",
        )

    msg = response.choices[0].message
    tool_calls_out: list[dict] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls_out.append({
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return {"content": msg.content or "", "tool_calls": tool_calls_out}


# ── Anthropic tool calling ─────────────────────────────────────────────────

async def _call_anthropic_with_tools(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> dict:
    import anthropic as anthropic_sdk

    kwargs: dict[str, Any] = {}
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url

    client = anthropic_sdk.AsyncAnthropic(**kwargs)

    system_text = ""
    chat_messages: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text = msg.get("content", "") if isinstance(msg.get("content"), str) else ""
        else:
            chat_messages.append(msg)

    # Convert OpenAI-style tool schema → Anthropic tool schema.
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": chat_messages,
        "max_tokens": max_tokens or 8192,
        "temperature": temperature,
        "tools": anthropic_tools,
    }
    if system_text:
        params["system"] = system_text

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: client.messages.create(**params),
            label=f"{resolved.provider_id}.messages.tools",
        )

    text_parts: list[str] = []
    tool_calls_out: list[dict] = []
    for block in response.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls_out.append({
                "id": getattr(block, "id", "") or "",
                "function": {
                    "name": getattr(block, "name", "") or "",
                    "arguments": json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                },
            })
    return {"content": "".join(text_parts), "tool_calls": tool_calls_out}


# ── Google Gemini tool calling ─────────────────────────────────────────────

async def _call_google_with_tools(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> dict:
    import google.generativeai as genai

    if resolved.api_key:
        genai.configure(api_key=resolved.api_key)

    # OpenAI-style tools → Gemini FunctionDeclaration list
    function_declarations = []
    for t in tools:
        fn = t.get("function", t)
        function_declarations.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    gemini_tools = [{"function_declarations": function_declarations}]

    system_instruction = _extract_system(messages) or None
    contents = _messages_to_gemini(messages)

    model = genai.GenerativeModel(
        resolved.model_id,
        system_instruction=system_instruction,
        tools=gemini_tools,
    )

    gen_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        gen_config["max_output_tokens"] = max_tokens

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: model.generate_content_async(contents, generation_config=gen_config),
            label=f"{resolved.provider_id}.generate.tools",
        )

    text_parts: list[str] = []
    tool_calls_out: list[dict] = []
    for cand in (response.candidates or []):
        for part in (cand.content.parts or []):
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                args = {}
                if hasattr(fc, "args"):
                    try:
                        args = dict(fc.args)
                    except Exception:
                        args = {}
                tool_calls_out.append({
                    "id": f"call_{len(tool_calls_out)}",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                })
            elif getattr(part, "text", None):
                text_parts.append(part.text)

    return {"content": "".join(text_parts), "tool_calls": tool_calls_out}


# ---------------------------------------------------------------------------
# OpenAI / compatible implementation
# ---------------------------------------------------------------------------

async def _call_openai_compatible(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> str:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {}
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key

    client = AsyncOpenAI(**kwargs)

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        params["max_tokens"] = max_tokens

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: client.chat.completions.create(**params),
            label=f"{resolved.provider_id}.chat",
        )
    return response.choices[0].message.content or ""


async def _stream_openai_compatible(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
):
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {}
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key

    client = AsyncOpenAI(**kwargs)

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens:
        params["max_tokens"] = max_tokens

    async with client.chat.completions.stream(**params) as stream:
        async for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

async def _call_anthropic(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> str:
    import anthropic as anthropic_sdk

    kwargs: dict[str, Any] = {}
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url

    client = anthropic_sdk.AsyncAnthropic(**kwargs)

    # Separate system prompt from messages
    system_text = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"] if isinstance(msg["content"], str) else ""
        else:
            chat_messages.append(msg)

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": chat_messages,
        "max_tokens": max_tokens or 8192,
        "temperature": temperature,
    }
    if system_text:
        params["system"] = system_text

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: client.messages.create(**params),
            label=f"{resolved.provider_id}.messages",
        )
    return response.content[0].text if response.content else ""


async def _stream_anthropic(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
):
    import anthropic as anthropic_sdk

    kwargs: dict[str, Any] = {}
    if resolved.api_key:
        kwargs["api_key"] = resolved.api_key
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url

    client = anthropic_sdk.AsyncAnthropic(**kwargs)

    system_text = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"] if isinstance(msg["content"], str) else ""
        else:
            chat_messages.append(msg)

    params: dict[str, Any] = {
        "model": resolved.model_id,
        "messages": chat_messages,
        "max_tokens": max_tokens or 8192,
        "temperature": temperature,
    }
    if system_text:
        params["system"] = system_text

    async with client.messages.stream(**params) as stream:
        async for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Google Gemini implementation
# ---------------------------------------------------------------------------

async def _call_google(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
) -> str:
    import google.generativeai as genai

    if resolved.api_key:
        genai.configure(api_key=resolved.api_key)

    model = genai.GenerativeModel(resolved.model_id)

    # Build Gemini content from messages
    contents = _messages_to_gemini(messages)
    system_instruction = _extract_system(messages)

    if system_instruction:
        model = genai.GenerativeModel(
            resolved.model_id,
            system_instruction=system_instruction,
        )

    gen_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        gen_config["max_output_tokens"] = max_tokens

    async with provider_semaphore(resolved.provider_id):
        response = await with_retry(
            lambda: model.generate_content_async(contents, generation_config=gen_config),
            label=f"{resolved.provider_id}.generate",
        )
    return response.text or ""


async def _stream_google(
    resolved: ResolvedModel,
    messages: list[dict],
    max_tokens: int | None,
    temperature: float,
):
    import google.generativeai as genai

    if resolved.api_key:
        genai.configure(api_key=resolved.api_key)

    system_instruction = _extract_system(messages)
    model = genai.GenerativeModel(
        resolved.model_id,
        system_instruction=system_instruction or None,
    )
    contents = _messages_to_gemini(messages)

    gen_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        gen_config["max_output_tokens"] = max_tokens

    async for chunk in await model.generate_content_async(
        contents,
        generation_config=gen_config,
        stream=True,
    ):
        if chunk.text:
            yield chunk.text


def _extract_system(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            return c if isinstance(c, str) else ""
    return ""


def _messages_to_gemini(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "parts": [{"text": content}]})
        else:
            result.append({"role": role, "parts": content})
    return result
