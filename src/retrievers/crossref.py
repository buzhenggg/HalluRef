"""CrossRef API 检索器"""

from __future__ import annotations

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class CrossRefRetriever(BaseRetriever):
    source_name = "crossref"

    def __init__(self, mailto: str | None = None, timeout: int = 15):
        # 节流由 Agent 2 入口锁统一控制
        super().__init__(timeout=timeout)
        self.base_url = "https://api.crossref.org/works"
        self.mailto = mailto

    def _params(self, **extra) -> dict:
        p = extra.copy()
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    def _parse_item(self, item: dict) -> RetrievedPaper:
        titles = item.get("title", [])
        title = titles[0] if titles else ""

        authors = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            authors.append(f"{given} {family}".strip())

        year = None
        date_parts = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or item.get("created", {}).get("date-parts")
        )
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        container = item.get("container-title", [])
        venue = container[0] if container else None

        return RetrievedPaper(
            source=self.source_name,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=item.get("DOI"),
            abstract=item.get("abstract"),
            url=item.get("URL"),
        )

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        params = self._params(
            **{"query.bibliographic": title, "rows": 5, "select": (
                "DOI,title,author,published-print,published-online,"
                "created,container-title,abstract,URL"
            )}
        )
        resp = await self._request_with_retry(
            "GET", self.base_url, params=params,
        )
        items = resp.json().get("message", {}).get("items", [])
        logger.debug(f"[CrossRef] title search → {len(items)} results")
        return [self._parse_item(it) for it in items]

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        params = self._params(
            **{"query.author": " ".join(authors[:2]), "rows": 5}
        )
        if year:
            params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"
        resp = await self._request_with_retry(
            "GET", self.base_url, params=params,
        )
        items = resp.json().get("message", {}).get("items", [])
        return [self._parse_item(it) for it in items]
