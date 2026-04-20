"""Image generation provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class ImageConfig:
    provider_id: str
    api_key: str
    base_url: str = ""
    model: str = ""


@dataclass
class ImageGenerationOptions:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    aspect_ratio: str = "16:9"
    style: str = ""


@dataclass
class ImageGenerationResult:
    url: str = ""
    base64: str = ""
    format: str = "png"


_DEFAULT_BASE_URLS = {
    "seedream": "https://api.siliconflow.cn/v1",
    "qwen-image": "https://dashscope.aliyuncs.com/api/v1",
}

_DEFAULT_MODELS = {
    "seedream": "black-forest-labs/FLUX.1-schnell",
    "qwen-image": "wanx2.1-t2i-turbo",
}

ASPECT_RATIO_DIMENSIONS = {
    "16:9": (1024, 576),
    "4:3": (1024, 768),
    "1:1": (1024, 1024),
    "9:16": (576, 1024),
    "3:4": (768, 1024),
}


async def generate_image(config: ImageConfig, options: ImageGenerationOptions) -> ImageGenerationResult:
    match config.provider_id:
        case "seedream":
            return await _seedream(config, options)
        case "qwen-image":
            return await _qwen_image(config, options)
        case _:
            raise ValueError(f"Unknown image provider: {config.provider_id}")


async def test_image_connectivity(config: ImageConfig) -> dict:
    try:
        await generate_image(config, ImageGenerationOptions(prompt="a simple red circle"))
        return {"success": True, "message": "Connection successful"}
    except Exception as e:
        return {"success": False, "message": str(e)}


async def _seedream(config: ImageConfig, options: ImageGenerationOptions) -> ImageGenerationResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["seedream"]
    model = config.model or _DEFAULT_MODELS["seedream"]
    w, h = ASPECT_RATIO_DIMENSIONS.get(options.aspect_ratio, (1024, 576))

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "prompt": options.prompt,
                "image_size": f"{w}x{h}",
                "num_inference_steps": 20,
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Seedream image error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    images = data.get("images") or data.get("data") or []
    if not images:
        raise RuntimeError(f"No images in response: {data}")

    url = images[0].get("url", "")
    b64 = images[0].get("b64_json", "")
    return ImageGenerationResult(url=url, base64=b64, format="png")


async def _qwen_image(config: ImageConfig, options: ImageGenerationOptions) -> ImageGenerationResult:
    base_url = config.base_url or _DEFAULT_BASE_URLS["qwen-image"]
    model = config.model or _DEFAULT_MODELS["qwen-image"]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/services/aigc/text2image/image-synthesis",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json={
                "model": model,
                "input": {"prompt": options.prompt},
                "parameters": {"size": f"{options.width}*{options.height}"},
            },
        )
    if not resp.is_success:
        raise RuntimeError(f"Qwen image error {resp.status_code}: {resp.text[:300]}")

    task_id = resp.json().get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError("Qwen image: no task_id")

    # Poll for result
    import asyncio
    for _ in range(30):
        await asyncio.sleep(3)
        async with httpx.AsyncClient(timeout=30) as client:
            poll = await client.get(
                f"{base_url}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        result = poll.json().get("output", {})
        if result.get("task_status") == "SUCCEEDED":
            url = (result.get("results") or [{}])[0].get("url", "")
            return ImageGenerationResult(url=url, format="png")
        if result.get("task_status") == "FAILED":
            raise RuntimeError(f"Qwen image task failed: {result}")

    raise RuntimeError("Qwen image: timed out waiting for result")
