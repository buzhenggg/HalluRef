"""SerpAPI Google Search 检索器

文档: https://serpapi.com/search-api
免费额度: 100 次/月 (注册即送)
端点: GET https://serpapi.com/search.json   (engine=google)
认证: api_key query 参数

注意: 与 Serper 类似, 返回的是 Google Web Search 通用网页结果,
       适合作为 Serper 不可用时的备选。
"""

from __future__ import annotations

import re

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class SerpApiRetriever(BaseRetriever):
    source_name = "serpapi"

    def __init__(self, api_key: str | None = None, timeout: int = 20, max_results: int = 5):
        super().__init__(timeout=timeout)
        self.base_url = "https://serpapi.com/search.json"
        self.api_key = api_key
        self.max_results = max_results

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _extract_year(text: str | None) -> int | None:
        if not text:
            return None
        m = re.search(r"(19|20)\d{2}", text)
        if not m:
            return None
        try:
            return int(m.group())
        except ValueError:
            return None

    def _parse_item(self, item: dict) -> RetrievedPaper:
        title = (item.get("title") or "").strip()
        link = item.get("link") or None
        snippet = item.get("snippet") or None
        date = item.get("date")

        year = self._extract_year(date) or self._extract_year(snippet)

        return RetrievedPaper(
            source=self.source_name,
            title=title,
            authors=[],
            year=year,
            venue=None,
            doi=None,
            abstract=snippet,
            url=link,
        )

    async def _query(self, q: str) -> list[RetrievedPaper]:
        if not q:
            return []
        params = {
            "engine": "google",
            "q": q,
            "api_key": self.api_key,
            "num": self.max_results,
        }
        resp = await self._request_with_retry("GET", self.base_url, params=params)
        data = resp.json()
        organic = data.get("organic_results") or []
        logger.debug(f"[SerpAPI] '{q[:40]}' → {len(organic)} results")
        return [self._parse_item(it) for it in organic[: self.max_results]]

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        return await self._query(title)

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        q = " ".join(authors[:2])
        if year:
            q += f" {year}"
        return await self._query(q)
