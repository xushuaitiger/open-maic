"""POST /api/classroom  and  GET /api/classroom?id=..."""

import uuid
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.errors import ApiException, ErrorCode, api_success
from core.storage.classroom_store import (
    is_valid_classroom_id,
    persist_classroom,
    read_classroom,
)

router = APIRouter()


class ClassroomCreateRequest(BaseModel):
    stage: dict[str, Any] = Field(..., description="Stage metadata, must include id")
    scenes: list[dict[str, Any]] = Field(..., min_length=1)


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("/classroom", status_code=201)
async def create_classroom(body: ClassroomCreateRequest, request: Request):
    classroom_id = body.stage.get("id") or str(uuid.uuid4())
    if not is_valid_classroom_id(classroom_id):
        raise ApiException(ErrorCode.INVALID_REQUEST, "Invalid classroom id in stage.id")

    data = {
        "id": classroom_id,
        "stage": {**body.stage, "id": classroom_id},
        "scenes": body.scenes,
    }
    result = await persist_classroom(data, _base_url(request))
    return api_success(result, status_code=201, request=request)


@router.get("/classroom")
async def get_classroom(request: Request, id: str = Query(..., min_length=1)):
    if not is_valid_classroom_id(id):
        raise ApiException(ErrorCode.INVALID_REQUEST, "Invalid classroom id")

    classroom = await read_classroom(id)
    if not classroom:
        raise ApiException(ErrorCode.NOT_FOUND, "Classroom not found", status_code=404)

    return api_success(classroom, request=request)
