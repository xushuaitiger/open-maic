"""POST /api/web-search — Tavily-backed web search."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.web_search.tavily import search_with_tavily

log = logging.getLogger("web_search")
router = APIRouter()


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    api_key: str = Field(default="", alias="apiKey")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.post("/web-search")
async def web_search(body: WebSearchRequest, request: Request):
    cfg = get_server_config()
    api_key = cfg.resolve_web_search_api_key(body.api_key)
    if not api_key:
        raise ApiException(
            ErrorCode.PROVIDER_ERROR,
            "Tavily API key not configured. Set TAVILY_API_KEY or configure it in server-providers.yml",
            status_code=400,
        )

    try:
        result = await search_with_tavily(body.query.strip(), api_key)
    except Exception as exc:
        log.error("Web search error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Web search failed: {exc}", status_code=502) from exc

    return api_success(
        {
            "answer": result.answer,
            "sources": result.sources,
            "context": result.context,
            "query": result.query,
            "responseTime": result.response_time,
        },
        request=request,
    )
