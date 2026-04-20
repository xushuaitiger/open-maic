"""POST /api/proxy-media — server-side CORS proxy for remote media."""

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.errors import ApiException, ErrorCode
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("proxy_media")
router = APIRouter()


class ProxyMediaRequest(BaseModel):
    url: str = Field(..., min_length=1)


@router.post("/proxy-media")
async def proxy_media(body: ProxyMediaRequest, request: Request):
    err = validate_url_for_ssrf(body.url)
    if err:
        raise ApiException(ErrorCode.INVALID_URL, err)

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            resp = await client.get(body.url)
    except Exception as exc:
        log.error("Proxy media error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, str(exc), status_code=502) from exc

    if 300 <= resp.status_code < 400:
        raise ApiException(ErrorCode.UPSTREAM_ERROR, "Redirects are not allowed", status_code=502)
    if not resp.is_success:
        raise ApiException(
            ErrorCode.UPSTREAM_ERROR, f"Upstream returned {resp.status_code}", status_code=502
        )

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers={
            "Content-Length": str(len(resp.content)),
            "Cache-Control": "private, max-age=3600",
        },
    )
