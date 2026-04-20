"""Shared API response models."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiSuccess(BaseModel, Generic[T]):
    success: bool = True
    data: T


class ApiError(BaseModel):
    success: bool = False
    error: str
    message: str
    details: str | None = None


def success(data: Any, status_code: int = 200) -> tuple[Any, int]:
    return {"success": True, "data": data}, status_code


def error(code: str, status: int, message: str, details: str = "") -> tuple[Any, int]:
    payload: dict = {"success": False, "error": code, "message": message}
    if details:
        payload["details"] = details
    return payload, status
