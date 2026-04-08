"""LLM 调用客户端 — 封装 OpenAI 兼容接口"""

from __future__ import annotations

import asyncio
import json
import os

from loguru import logger
from openai import AsyncOpenAI


class LLMClient:
    """异步 LLM 客户端, 支持 OpenAI 兼容 API

    内置进程级并发限流 (Semaphore), 避免同时打爆 LLM 配额。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_concurrent: int = 4,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
        )
        self._sem = asyncio.Semaphore(max_concurrent)

    async def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> str:
        """发送单轮对话, 返回助手回复文本"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._sem:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
            )
        content = resp.choices[0].message.content or ""
        logger.debug(f"[LLM] {len(content)} chars response")
        return content

    async def chat_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> dict | list:
        """发送对话并解析 JSON 返回"""
        text = await self.chat(prompt, system, temperature)
        # 尝试提取 JSON 块
        text = text.strip()
        if text.startswith("```"):
            # 去掉 markdown 代码块
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
