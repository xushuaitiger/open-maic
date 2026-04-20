"""POST /api/parse-pdf — extract text + images from a PDF."""

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.config import get_server_config
from app.errors import ApiException, ErrorCode, api_success
from core.providers.pdf import PDFConfig, parse_pdf
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("parse_pdf")
router = APIRouter()


@router.post("/parse-pdf")
async def parse_pdf_endpoint(
    request: Request,
    pdf: UploadFile = File(...),
    providerId: str = Form("unpdf"),
    apiKey: str = Form(""),
    baseUrl: str = Form(""),
):
    if not pdf or not pdf.filename:
        raise ApiException(ErrorCode.MISSING_REQUIRED_FIELD, "pdf file is required")

    if baseUrl:
        err = validate_url_for_ssrf(baseUrl)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)

    cfg = get_server_config()
    resolved_key = apiKey or cfg.resolve_pdf_api_key(providerId, apiKey)
    resolved_url = baseUrl or cfg.resolve_pdf_base_url(providerId)

    try:
        pdf_bytes = await pdf.read()
        result = await parse_pdf(
            PDFConfig(provider_id=providerId, api_key=resolved_key, base_url=resolved_url),
            pdf_bytes,
        )
    except Exception as exc:
        log.error("PDF parse error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.PARSE_FAILED, f"Failed to parse PDF: {exc}", status_code=502) from exc

    return api_success(
        {
            "text": result.text,
            "images": result.images,
            "pageCount": result.page_count,
        },
        request=request,
    )
