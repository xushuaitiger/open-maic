"""GET /api/health"""

from fastapi import APIRouter
from app.config import get_server_config

router = APIRouter()

VERSION = "0.1.0"


@router.get("/health")
async def health():
    cfg = get_server_config()
    return {
        "success": True,
        "data": {
            "status": "ok",
            "version": VERSION,
            "capabilities": {
                "webSearch": bool(cfg.get_web_search_providers()),
                "imageGeneration": bool(cfg.get_image_providers()),
                "videoGeneration": bool(cfg.get_video_providers()),
                "tts": bool(cfg.get_tts_providers()),
            },
        },
    }
