"""LLM 调用层：只依赖 OpenAI 兼容接口，因此后端可以在
LMDeploy / vLLM / Ollama / 任何兼容服务之间自由切换，不需要改业务代码。
任何异常都返回 None，由上层走兜底话术，绝不让评测请求 500。
"""
import asyncio

import httpx

from .config import settings


async def chat(messages: list[dict], max_tokens: int | None = None,
               temperature: float | None = None, timeout: float | None = None) -> str | None:
    if settings.llm_backend != "openai":
        return None

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens or settings.max_new_tokens,
        "temperature": settings.temperature if temperature is None else temperature,
        "top_p": settings.top_p,
        "stream": False,
    }
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout or settings.request_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                if text:
                    return text
                print("[llm] 返回空内容", flush=True)
        except Exception as exc:
            print(f"[llm] 第 {attempt + 1} 次调用失败: {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(0.4)
    return None


async def health() -> dict:
    if settings.llm_backend != "openai":
        return {"backend": "none", "ready": False}
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {settings.llm_api_key}"})
            return {"backend": "openai", "ready": resp.status_code == 200,
                    "base_url": settings.llm_base_url}
    except Exception as exc:
        return {"backend": "openai", "ready": False, "error": str(exc)}
