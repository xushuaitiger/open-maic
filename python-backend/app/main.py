"""
OpenMAIC Python Backend — FastAPI application entry point.

Run:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_server_config, get_settings
from app.errors import (
    ApiException,
    api_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.middleware import AccessLogMiddleware, RequestIdMiddleware

# ── Route imports ────────────────────────────────────────────────────────────
from app.api.health import router as health_router
from app.api.server_providers import router as server_providers_router
from app.api.classroom import router as classroom_router
from app.api.generate_classroom import router as generate_classroom_router
from app.api.chat import router as chat_router
from app.api.pbl_chat import router as pbl_chat_router
from app.api.tts import router as tts_router
from app.api.transcription import router as transcription_router
from app.api.azure_voices import router as azure_voices_router
from app.api.parse_pdf import router as parse_pdf_router
from app.api.web_search import router as web_search_router
from app.api.quiz_grade import router as quiz_grade_router
from app.api.proxy_media import router as proxy_media_router
from app.api.classroom_media import router as classroom_media_router
from app.api.verify_model import router as verify_model_router
from app.api.verify_pdf_provider import router as verify_pdf_router
from app.api.verify_image_provider import router as verify_image_router
from app.api.verify_video_provider import router as verify_video_router
from app.api.generate.scene_outlines_stream import router as outlines_stream_router
from app.api.generate.scene_content import router as scene_content_router
from app.api.generate.scene_actions import router as scene_actions_router
from app.api.generate.agent_profiles import router as agent_profiles_router
from app.api.generate.image import router as image_gen_router
from app.api.generate.video import router as video_gen_router

from core.security.secret_masking import mask_secret

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app")

# ── App ──────────────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="OpenMAIC Backend",
    description="AI-powered interactive classroom generation API",
    version="0.1.0",
)

# Middleware order matters: outermost first.  Access log must wrap request id
# so it can read the assigned id; CORS sits at the very edge.
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)

# ── Exception handlers ───────────────────────────────────────────────────────
app.add_exception_handler(ApiException, api_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── Mount all routers under /api ─────────────────────────────────────────────
API_PREFIX = "/api"

for router in (
    health_router,
    server_providers_router,
    classroom_router,
    generate_classroom_router,
    chat_router,
    pbl_chat_router,
    tts_router,
    transcription_router,
    azure_voices_router,
    parse_pdf_router,
    web_search_router,
    quiz_grade_router,
    proxy_media_router,
    classroom_media_router,
    verify_model_router,
    verify_pdf_router,
    verify_image_router,
    verify_video_router,
    outlines_stream_router,
    scene_content_router,
    scene_actions_router,
    agent_profiles_router,
    image_gen_router,
    video_gen_router,
):
    app.include_router(router, prefix=API_PREFIX)


@app.get("/")
async def root():
    return {"message": "OpenMAIC Backend is running", "docs": "/docs"}


# ── Startup diagnostics ──────────────────────────────────────────────────────

@app.on_event("startup")
async def _log_provider_summary() -> None:
    """Log a one-shot summary of which providers are configured.

    API keys are masked.  Useful for spotting wiring bugs without leaking
    secrets to log aggregators.
    """
    cfg = get_server_config()
    log.info("OpenMAIC backend starting (default_model=%s)", settings.default_model)

    sections = (
        ("LLM",    cfg._llm),         # noqa: SLF001 — internal but stable shape
        ("TTS",    cfg._tts),
        ("ASR",    cfg._asr),
        ("PDF",    cfg._pdf),
        ("Image",  cfg._image),
        ("Video",  cfg._video),
        ("Search", cfg._web_search),
    )
    for label, mapping in sections:
        configured = []
        for pid, entry in mapping.items():
            if entry.api_key:
                base_url = entry.base_url or "<default>"
                configured.append(f"{pid}(key={mask_secret(entry.api_key)} base={base_url})")
        if configured:
            log.info("  %-6s providers: %s", label, ", ".join(configured))
        else:
            log.info("  %-6s providers: (none configured)", label)

    log.info(
        "Config priority: client headers > .env / env vars > server-providers.yml > defaults"
    )
