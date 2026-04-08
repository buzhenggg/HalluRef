"""Serper Google Search API 检索器

文档: https://serper.dev/
免费额度: 注册即送 2500 次查询
端点: POST https://google.serper.dev/search   (Google Web Search)
认证: X-API-KEY header

注意: 使用 Google Web Search (而非 Google Scholar), 返回结构是通用网页结果
       (title / link / snippet / date), 不含结构化的作者/期刊字段。
       适合用作"是否存在该论文相关网页"的存在性证据。
"""

from __future__ import annotations

import json
import re

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class SerperRetriever(BaseRetriever):
    source_name = "serper"

    def __init__(self, api_key: str | None = None, timeout: int = 15, max_results: int = 5):
        super().__init__(timeout=timeout)
        self.base_url = "https://google.serper.dev/search"
        self.api_key = api_key
        self.max_results = max_results

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

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

        # date 字段如 "Jun 12, 2017" / 直接 "2017"; 否则从 snippet 提
        year = self._extract_year(item.get("date")) or self._extract_year(snippet)

        return RetrievedPaper(
            source=self.source_name,
            title=title,
            authors=[],         # Google Web Search 不返回结构化作者
            year=year,
            venue=None,
            doi=None,
            abstract=snippet,
            url=link,
        )

    async def _query(self, q: str) -> list[RetrievedPaper]:
        if not q:
            return []
        payload = {"q": q, "num": self.max_results}
        resp = await self._request_with_retry(
            "POST",
            self.base_url,
            content=json.dumps(payload),
            headers=self._headers(),
        )
        data = resp.json()
        organic = data.get("organic") or []
        logger.debug(f"[Serper] '{q[:40]}' → {len(organic)} results")
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
