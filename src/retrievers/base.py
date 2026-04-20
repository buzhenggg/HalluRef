"""检索器基类 — 含重试、限流机制"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import httpx
from loguru import logger

from src.models.schemas import RetrievedPaper


class BaseRetriever(ABC):
    """学术论文检索器基类

    内置:
    - 指数退避重试 (针对 429 / 5xx)
    - 请求间隔限流 (避免短时间高并发触发限流)
    """

    source_name: str = "unknown"

    def __init__(
        self,
        timeout: int = 15,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        request_interval: float = 0.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_interval = request_interval
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0
        self.last_search_debug: str = ""

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def is_configured(self) -> bool:
        """检索器是否配置完整, 可用于发起请求。

        默认无强制配置项, 子类可覆盖 (如 Serper 必须有 api_key)。
        """
        return True

    async def _rate_limit(self):
        """请求间隔限流"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """带重试的 HTTP 请求

        对 429 (限流) 和 5xx (服务器错误) 自动重试, 指数退避。
        """
        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            await self._rate_limit()
            try:
                resp = await client.request(method, url, **kwargs)
                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"[{self.source_name}] HTTP {resp.status_code}, "
                        f"retry {attempt + 1}/{self.max_retries} after {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"[{self.source_name}] request error: {e}, "
                        f"retry {attempt + 1}/{self.max_retries} after {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        raise last_exc or RuntimeError(
            f"[{self.source_name}] max retries exceeded for {url}"
        )

    @abstractmethod
    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        """按标题检索论文"""

    @abstractmethod
    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        """按作者+年份检索论文"""

    async def search(
        self,
        title: str = "",
        authors: list[str] | None = None,
        year: int | None = None,
    ) -> list[RetrievedPaper]:
        """综合检索: 先按标题, 无结果则按作者+年份"""
        results: list[RetrievedPaper] = []
        self.last_search_debug = ""

        if title:
            try:
                results = await self.search_by_title(title)
            except Exception as e:
                self.last_search_debug = f"error ({e})"
                logger.warning(f"[{self.source_name}] title search failed: {e}")

        if not results and authors:
            try:
                results = await self.search_by_author_year(authors, year)
            except Exception as e:
                self.last_search_debug = f"error ({e})"
                logger.warning(f"[{self.source_name}] author search failed: {e}")

        if not self.last_search_debug:
            self.last_search_debug = f"{len(results)} candidates"

        return results
