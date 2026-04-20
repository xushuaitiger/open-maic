"""GET /api/classroom-media/{classroom_id}/{path...} — serve stored media files."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.errors import ApiException, ErrorCode
from core.storage.classroom_store import get_classroom_media_path, is_valid_classroom_id

router = APIRouter()

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
}


@router.get("/classroom-media/{classroom_id}/{path:path}")
async def classroom_media(classroom_id: str, path: str):
    if not is_valid_classroom_id(classroom_id):
        raise ApiException(ErrorCode.INVALID_REQUEST, "Invalid classroom id")

    path_segments = [p for p in path.split("/") if p]
    real_path = await get_classroom_media_path(classroom_id, path_segments)
    if not real_path:
        raise ApiException(ErrorCode.NOT_FOUND, "Media not found", status_code=404)

    content_type = _MIME_TYPES.get(real_path.suffix.lower(), "application/octet-stream")

    return FileResponse(
        real_path,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
