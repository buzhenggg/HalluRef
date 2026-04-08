"""OpenAlex API 检索器"""

from __future__ import annotations

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class OpenAlexRetriever(BaseRetriever):
    source_name = "openalex"

    def __init__(self, mailto: str | None = None, timeout: int = 15):
        # 节流由 Agent 2 入口锁统一控制, 此处不再 per-instance 限速
        super().__init__(timeout=timeout)
        self.base_url = "https://api.openalex.org/works"
        self.mailto = mailto

    def _parse_work(self, work: dict) -> RetrievedPaper:
        title = work.get("title", "") or ""

        authors = []
        for authorship in work.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)

        year = work.get("publication_year")

        source = work.get("primary_location", {}).get("source") or {}
        venue = source.get("display_name")

        doi_url = work.get("doi") or ""
        doi = doi_url.replace("https://doi.org/", "") if doi_url else None

        abstract = self._rebuild_abstract(work.get("abstract_inverted_index"))

        return RetrievedPaper(
            source=self.source_name,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            abstract=abstract,
            url=work.get("id"),
        )

    @staticmethod
    def _rebuild_abstract(inverted_index: dict | None) -> str | None:
        """从 OpenAlex 的倒排索引重建摘要文本"""
        if not inverted_index:
            return None
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        params: dict = {
            "search": title,
            "per_page": 5,
            "select": "id,doi,title,authorships,publication_year,primary_location,abstract_inverted_index",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        resp = await self._request_with_retry(
            "GET", self.base_url, params=params,
        )
        results = resp.json().get("results", [])
        logger.debug(f"[OpenAlex] title search → {len(results)} results")
        return [self._parse_work(w) for w in results]

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        query = " ".join(authors[:2])
        params: dict = {"search": query, "per_page": 5}
        if year:
            params["filter"] = f"publication_year:{year}"
        if self.mailto:
            params["mailto"] = self.mailto
        resp = await self._request_with_retry(
            "GET", self.base_url, params=params,
        )
        results = resp.json().get("results", [])
        return [self._parse_work(w) for w in results]
