"""PDF parsing provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class PDFConfig:
    provider_id: str
    api_key: str = ""
    base_url: str = ""


@dataclass
class ParsedPDFContent:
    text: str
    images: list[str] = field(default_factory=list)   # base64 strings
    page_count: int = 0


async def parse_pdf(config: PDFConfig, pdf_bytes: bytes) -> ParsedPDFContent:
    match config.provider_id:
        case "unpdf":
            return await _unpdf(pdf_bytes)
        case "mineru":
            return await _mineru(config, pdf_bytes)
        case _:
            raise ValueError(f"Unknown PDF provider: {config.provider_id}")


async def _unpdf(pdf_bytes: bytes) -> ParsedPDFContent:
    """Parse PDF using pypdf (built-in fallback, no API key needed)."""
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return ParsedPDFContent(
            text="\n\n".join(pages),
            images=[],
            page_count=len(reader.pages),
        )
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {e}") from e


async def _mineru(config: PDFConfig, pdf_bytes: bytes) -> ParsedPDFContent:
    """Parse PDF using MinerU API (better OCR, tables, formulas)."""
    base_url = config.base_url
    if not base_url:
        raise ValueError("MinerU requires a base_url")

    headers: dict = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/api/v1/extract",
            headers=headers,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
        )
    if not resp.is_success:
        raise RuntimeError(f"MinerU error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    return ParsedPDFContent(
        text=data.get("markdown", data.get("text", "")),
        images=data.get("images", []),
        page_count=data.get("page_count", 0),
    )
