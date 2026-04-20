"""Tavily web search integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

_TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class SearchResult:
    query: str
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    context: str = ""
    response_time: float = 0.0


async def search_with_tavily(query: str, api_key: str, max_results: int = 5) -> SearchResult:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": max_results,
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Tavily error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    sources = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in data.get("results", [])
    ]
    context = "\n\n".join(
        f"[{i+1}] {s['title']}\n{s['content']}" for i, s in enumerate(sources)
    )

    return SearchResult(
        query=data.get("query", query),
        answer=data.get("answer", ""),
        sources=sources,
        context=context,
        response_time=data.get("response_time", 0.0),
    )
