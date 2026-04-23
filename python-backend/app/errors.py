"""
Unified error handling for the OpenMAIC backend.

All API endpoints should:
  • Return success responses via `api_success(data, status=200)` (or just plain dict)
  • Raise `ApiException` for expected business errors — they are translated to
    a consistent JSON envelope with the proper HTTP status code
  • Let unexpected exceptions bubble up — `unhandled_exception_handler`
    converts them to a 500 with a stable `INTERNAL_ERROR` code while still
    logging the full traceback

Envelope format (matches the original Next.js /lib/api/response.ts):

  Success:
    { "success": true, "data": <anything> }

  Error:
    {
      "success": false,
      "error":   "ERROR_CODE",       # stable machine identifier
      "message": "Human readable",
      "details": "optional extra info",
      "requestId": "<request id>"    # always included for traceability
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("errors")


# ── Error code constants ────────────────────────────────────────────────────

class ErrorCode:
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_URL = "INVALID_URL"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PARSE_FAILED = "PARSE_FAILED"
    LLM_EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiException(Exception):
    """Raise this for any business / validation error you want surfaced as JSON."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _request_id_of(request: Request | None) -> str:
    if request is None:
        return ""
    return getattr(request.state, "request_id", "") or ""


def api_success(data: Any, *, status_code: int = 200, request: Request | None = None) -> JSONResponse:
    """Build a standard success JSONResponse.

    Matches the TypeScript apiSuccess() which spreads fields flat:
      { success: true, ...data }
    NOT nested under a 'data' key.
    """
    if isinstance(data, dict):
        payload: dict[str, Any] = {"success": True, **data}
    else:
        payload = {"success": True, "data": data}
    rid = _request_id_of(request)
    if rid:
        payload["requestId"] = rid
    return JSONResponse(payload, status_code=status_code)


def api_error(
    code: str,
    message: str,
    *,
    status_code: int = 400,
    details: str | None = None,
    request: Request | None = None,
) -> JSONResponse:
    """Build a standard error JSONResponse."""
    payload: dict[str, Any] = {
        "success": False,
        "error": code,
        "message": message,
    }
    if details:
        payload["details"] = details
    rid = _request_id_of(request)
    if rid:
        payload["requestId"] = rid
    return JSONResponse(payload, status_code=status_code)


# ── FastAPI exception handlers (registered in main.py) ──────────────────────

async def api_exception_handler(request: Request, exc: ApiException) -> JSONResponse:
    return api_error(
        exc.code,
        exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request=request,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = ErrorCode.INVALID_REQUEST
    if exc.status_code == 404:
        code = ErrorCode.NOT_FOUND
    elif exc.status_code == 401:
        code = ErrorCode.UNAUTHORIZED
    elif exc.status_code == 403:
        code = ErrorCode.FORBIDDEN
    elif exc.status_code >= 500:
        code = ErrorCode.INTERNAL_ERROR
    return api_error(code, str(exc.detail), status_code=exc.status_code, request=request)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    # Compact pydantic error rendering
    issues = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []) if x not in ("body", "query", "path"))
        issues.append(f"{loc}: {err.get('msg', '')}".strip(": "))
    details = "; ".join(issues)[:1000]
    return api_error(
        ErrorCode.INVALID_REQUEST,
        "Request body validation failed",
        status_code=422,
        details=details,
        request=request,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id_of(request)
    log.error(
        "Unhandled exception (rid=%s, path=%s): %s",
        rid, request.url.path, exc,
        exc_info=True,
    )
    return api_error(
        ErrorCode.INTERNAL_ERROR,
        "Internal server error",
        status_code=500,
        details=type(exc).__name__,
        request=request,
    )
