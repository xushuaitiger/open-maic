"""
POST /api/generate-classroom         — submit a generation job
GET  /api/generate-classroom/{id}    — poll job status
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from nanoid import generate as nanoid

from app.errors import ApiException, ErrorCode, api_success
from core.generation.job_runner import run_classroom_job
from core.storage.job_store import create_job, is_valid_job_id, read_job

router = APIRouter()


class GenerateClassroomRequest(BaseModel):
    requirement: str = Field(..., min_length=1)
    pdf_content: dict[str, Any] | None = None
    language: Literal["zh-CN", "en-US"] | None = None
    enable_web_search: bool = False
    enable_image_generation: bool = False
    enable_video_generation: bool = False
    enable_tts: bool = False
    agent_mode: Literal["default", "generate"] = "default"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("/generate-classroom", status_code=202)
async def submit_generate_classroom(body: GenerateClassroomRequest, request: Request):
    base = _base_url(request)
    job_id = nanoid(10)
    input_data = body.model_dump()
    job = await create_job(job_id, input_data)

    # Fire-and-forget background task — see TODO in job_runner about durability.
    run_classroom_job(job_id, input_data, base)

    return api_success(
        {
            "jobId": job_id,
            "status": job["status"],
            "step": job["step"],
            "message": job["message"],
            "pollUrl": f"{base}/api/generate-classroom/{job_id}",
            "pollIntervalMs": 5000,
        },
        status_code=202,
        request=request,
    )


@router.get("/generate-classroom/{job_id}")
async def get_generate_classroom_job(job_id: str, request: Request):
    if not is_valid_job_id(job_id):
        raise ApiException(ErrorCode.INVALID_REQUEST, "Invalid job id")

    job = await read_job(job_id)
    if not job:
        raise ApiException(ErrorCode.NOT_FOUND, "Job not found", status_code=404)

    base = _base_url(request)
    return api_success(
        {
            "jobId": job["id"],
            "status": job["status"],
            "step": job.get("step"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "pollUrl": f"{base}/api/generate-classroom/{job_id}",
            "pollIntervalMs": 5000,
            "scenesGenerated": job.get("scenesGenerated", 0),
            "totalScenes": job.get("totalScenes"),
            "result": job.get("result"),
            "error": job.get("error"),
            "done": job["status"] in ("succeeded", "failed"),
        },
        request=request,
    )
