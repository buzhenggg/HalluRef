"""SerpAPI Google / Google Scholar Search retriever.

文档: https://serpapi.com/search-api
免费额度: 100 次/月 (注册即送)
端点: GET https://serpapi.com/search.json
认证: api_key query 参数

注意: 当前级联编排只注入 google_scholar engine；google engine 保留为独立检索器能力。
"""

from __future__ import annotations

import re
import asyncio

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever
from src.retrievers.page_metadata import fetch_page_metadata


class SerpApiRetriever(BaseRetriever):
    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 20,
        max_results: int = 5,
        enrich_links: bool = True,
        search_type: str = "web",
        page_fetch_proxy: str | None = None,
    ):
        super().__init__(timeout=timeout)
        self.search_type = search_type
        self.source_name = "serpapi_scholar" if search_type == "scholar" else "serpapi"
        self.base_url = "https://serpapi.com/search.json"
        self.api_key = api_key
        self.max_results = max_results
        self.enrich_links = enrich_links
        self.page_fetch_proxy = page_fetch_proxy

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

    async def _enrich_result(self, paper: RetrievedPaper) -> RetrievedPaper:
        if not self.enrich_links or not paper.url:
            return paper
        enriched = await fetch_page_metadata(
            paper.url,
            timeout=self.timeout,
            proxy=self.page_fetch_proxy,
        )
        if enriched is None:
            return paper
        enriched.source = self.source_name
        if not enriched.abstract:
            enriched.abstract = paper.abstract
        if not enriched.year:
            enriched.year = paper.year
        if not enriched.url:
            enriched.url = paper.url
        return enriched

    async def _query(self, q: str) -> list[RetrievedPaper]:
        if not q:
            return []
        params = {
            "engine": "google_scholar" if self.search_type == "scholar" else "google",
            "q": q,
            "api_key": self.api_key,
            "num": self.max_results,
        }
        resp = await self._request_with_retry("GET", self.base_url, params=params)
        data = resp.json()
        organic = data.get("organic_results") or []
        logger.debug(f"[SerpAPI] '{q[:40]}' → {len(organic)} results")
        papers = [self._parse_item(it) for it in organic[: self.max_results]]
        return await asyncio.gather(*[self._enrich_result(p) for p in papers])

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        return await self._query(title)

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        q = " ".join(authors[:2])
        if year:
            q += f" {year}"
        return await self._query(q)
