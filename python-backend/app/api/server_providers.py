"""GET /api/server-providers"""

from fastapi import APIRouter
from app.config import get_server_config

router = APIRouter()


@router.get("/server-providers")
async def server_providers():
    cfg = get_server_config()
    return {
        "success": True,
        "data": {
            "providers": cfg.get_llm_providers(),
            "tts": cfg.get_tts_providers(),
            "asr": cfg.get_asr_providers(),
            "pdf": cfg.get_pdf_providers(),
            "image": cfg.get_image_providers(),
            "video": cfg.get_video_providers(),
            "webSearch": cfg.get_web_search_providers(),
        },
    }
