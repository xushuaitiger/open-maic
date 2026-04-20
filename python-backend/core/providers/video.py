"""Video generation provider implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass
class VideoConfig:
    provider_id: str
    api_key: str
    base_url: str = ""
    model: str = ""


@dataclass
class VideoGenerationOptions:
    prompt: str
    duration: int = 5
    aspect_ratio: str = "16:9"
    resolution: str = "1080p"


@dataclass
class VideoGenerationResult:
    url: str = ""
    format: str = "mp4"


_DEFAULT_BASE_URLS = {
    "seedance": "https://api.siliconflow.cn/v1",
    "kling": "https://api.klingai.com/v1",
}

_DEFAULT_MODELS = {
    "seedance": "Pro/Wan/Seedance-1-lite",
    "kling": "kling-v1",
}


async def generate_video(config: VideoConfig, options: VideoGenerationOptions) -> VideoGenerationResult:
    match config.provider_id:
        case "seedance":
            return await _seedance(config, options)
        case "kling":
            return await _kling(config, options)
        case _:
            raise ValueError(f"Unknown video provider: {config.provider_id}")


async def test_video_connectivity(config: VideoConfig) -> dict:
    try:
        await generate_video(config, VideoGenerationOptions(prompt="a simple animation"))
        return {"success": True, "message": "Connection successful"}
    except Exception as e:
        return {"success": False, "message": str(e)}


async def _seedance(config: VideoConfig, options: VideoGenerationOptions) -> VideoGenerationResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["seedance"]
    model = config.model or _DEFAULT_MODELS["seedance"]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/video/submit",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "prompt": options.prompt,
                "image_size": "1280x720" if options.aspect_ratio == "16:9" else "720x1280",
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Seedance video error {resp.status_code}: {resp.text[:300]}")

    request_id = resp.json().get("requestId")
    if not request_id:
        raise RuntimeError("Seedance video: no requestId")

    for _ in range(60):
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=30) as client:
            poll = await client.get(
                f"{base_url}/video/results",
                headers={"Authorization": f"Bearer {config.api_key}"},
                params={"requestId": request_id},
            )
        result = poll.json()
        status = result.get("status")
        if status == "Succeed":
            url = ((result.get("results") or {}).get("videos") or [{}])[0].get("url", "")
            return VideoGenerationResult(url=url)
        if status in ("Failed", "Error"):
            raise RuntimeError(f"Seedance video failed: {result}")

    raise RuntimeError("Seedance video: timed out")


async def _kling(config: VideoConfig, options: VideoGenerationOptions) -> VideoGenerationResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["kling"]
    model = config.model or _DEFAULT_MODELS["kling"]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/videos/text2video",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model_name": model,
                "prompt": options.prompt,
                "duration": str(options.duration),
                "aspect_ratio": options.aspect_ratio,
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Kling video error {resp.status_code}: {resp.text[:300]}")

    task_id = resp.json().get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError("Kling video: no task_id")

    for _ in range(60):
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=30) as client:
            poll = await client.get(
                f"{base_url}/videos/text2video/{task_id}",
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        result = poll.json().get("data", {})
        if result.get("task_status") == "succeed":
            url = (result.get("task_result") or {}).get("videos", [{}])[0].get("url", "")
            return VideoGenerationResult(url=url)
        if result.get("task_status") in ("failed", "error"):
            raise RuntimeError(f"Kling video failed: {result}")

    raise RuntimeError("Kling video: timed out")
