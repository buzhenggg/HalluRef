"""Semantic Scholar API 检索器

文档: https://api.semanticscholar.org/api-docs/graph
免费额度: 无 key 1 req/s, 申请 key 后 10 req/s
覆盖: 2 亿+论文, 含原生 abstract / TLDR
"""

from __future__ import annotations

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class SemanticScholarRetriever(BaseRetriever):
    source_name = "semantic_scholar"

    _FIELDS = "title,authors,year,venue,externalIds,abstract,tldr,url"

    def __init__(self, api_key: str | None = None, timeout: int = 15):
        # 节流由 Agent 2 入口锁统一控制
        super().__init__(timeout=timeout)
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper"
        self.api_key = api_key

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def is_configured(self) -> bool:
        return bool(str(self.api_key or "").strip())

    def _parse_paper(self, p: dict) -> RetrievedPaper:
        authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]

        external_ids = p.get("externalIds") or {}
        doi = external_ids.get("DOI")

        # 优先用原生 abstract, 否则用 TLDR
        abstract = p.get("abstract")
        if not abstract and p.get("tldr"):
            abstract = (p.get("tldr") or {}).get("text")

        return RetrievedPaper(
            source=self.source_name,
            title=p.get("title") or "",
            authors=authors,
            year=p.get("year"),
            venue=p.get("venue") or None,
            doi=doi,
            abstract=abstract,
            url=p.get("url"),
        )

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        params = {"query": title, "limit": 5, "fields": self._FIELDS}
        resp = await self._request_with_retry(
            "GET", f"{self.base_url}/search", params=params, headers=self._headers(),
        )
        data = resp.json().get("data") or []
        logger.debug(f"[SemanticScholar] title search → {len(data)} results")
        return [self._parse_paper(p) for p in data]

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        query = " ".join(authors[:2])
        params: dict = {"query": query, "limit": 5, "fields": self._FIELDS}
        if year:
            params["year"] = str(year)
        resp = await self._request_with_retry(
            "GET", f"{self.base_url}/search", params=params, headers=self._headers(),
        )
        data = resp.json().get("data") or []
        return [self._parse_paper(p) for p in data]
