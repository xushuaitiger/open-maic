"""
Cross-cutting HTTP middleware:

  • RequestIdMiddleware  — assign / propagate `X-Request-Id` and stash it on
    `request.state.request_id` so every log line and error envelope can
    reference the same identifier.

  • AccessLogMiddleware  — single concise access log per request:
        [INFO] access rid=ab12cd method=POST path=/api/chat status=200 ms=482
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-Id"

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(self.HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[self.HEADER] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            rid = getattr(request.state, "request_id", "") or ""
            log.info(
                "access rid=%s method=%s path=%s status=%s ms=%d",
                rid,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            rid = getattr(request.state, "request_id", "") or ""
            log.exception(
                "access rid=%s method=%s path=%s status=ERR ms=%d",
                rid, request.method, request.url.path, elapsed_ms,
            )
            raise


class RequestIdLogFilter(logging.Filter):
    """Optional logging filter — leaves a placeholder when there is no request scope."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True
