"""POST /api/quiz-grade — LLM-graded short-answer quiz scoring."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import get_server_config, get_settings
from app.errors import ApiException, ErrorCode, api_success
from core.generation.prompt_strings import quiz_grade_system, quiz_grade_user
from core.providers.json_utils import parse_llm_json
from core.providers.llm import call_llm, resolve_model
from core.security.ssrf_guard import validate_url_for_ssrf

log = logging.getLogger("quiz_grade")
router = APIRouter()


class QuizGradeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_answer: str = Field(..., min_length=1, alias="userAnswer")
    points: int = Field(default=1, ge=1, le=100)
    comment_prompt: str = Field(default="", alias="commentPrompt")
    language: str = "zh-CN"

    model_config = {"populate_by_name": True, "extra": "ignore"}


def _resolve_model_from_headers(request: Request, cfg, settings):
    model_str = request.headers.get("x-model") or settings.default_model
    api_key = request.headers.get("x-api-key", "")
    base_url = request.headers.get("x-base-url", "")
    if base_url:
        err = validate_url_for_ssrf(base_url)
        if err:
            raise ApiException(ErrorCode.INVALID_URL, err)
    return resolve_model(model_str, api_key, base_url, cfg)


@router.post("/quiz-grade")
async def quiz_grade(body: QuizGradeRequest, request: Request):
    cfg = get_server_config()
    settings = get_settings()
    resolved = _resolve_model_from_headers(request, cfg, settings)

    system = quiz_grade_system(body.points, body.language)
    user = quiz_grade_user(body.question, body.user_answer, body.points, body.comment_prompt, body.language)

    try:
        raw = await call_llm(
            resolved,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=256,
        )
        result = parse_llm_json(raw, expect="object")
    except Exception as exc:
        log.error("Quiz grade error (rid=%s): %s", getattr(request.state, "request_id", "-"), exc)
        raise ApiException(ErrorCode.UPSTREAM_ERROR, f"Quiz grading failed: {exc}", status_code=502) from exc

    if not isinstance(result, dict):
        raise ApiException(ErrorCode.PARSE_FAILED, "Grader returned non-object JSON", status_code=502)

    return api_success(
        {
            "score": int(result.get("score", 0)),
            "comment": str(result.get("comment", "")),
        },
        request=request,
    )
