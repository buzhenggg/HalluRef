"""arXiv API 检索器

文档: https://info.arxiv.org/help/api/user-manual.html
免费额度: 无限 (建议 3s 间隔)
覆盖: CS / 物理 / 数学 等领域预印本, 含原生 abstract
返回格式: Atom XML
"""

from __future__ import annotations

import asyncio
import re
from xml.etree import ElementTree as ET

from loguru import logger

from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


# Atom + arXiv 命名空间
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivRetriever(BaseRetriever):
    source_name = "arxiv"
    _global_lock: asyncio.Lock | None = None
    _global_last_request_time: float = 0.0
    _global_request_interval: float = 3.5

    def __init__(self, timeout: int = 15, max_results: int = 5):
        # 节流由 Agent 2 入口锁统一控制
        super().__init__(timeout=timeout)
        self.base_url = "https://export.arxiv.org/api/query"
        self.max_results = max_results

    @classmethod
    def _get_global_lock(cls) -> asyncio.Lock:
        if cls._global_lock is None:
            cls._global_lock = asyncio.Lock()
        return cls._global_lock

    async def _rate_limit(self):
        lock = self._get_global_lock()
        async with lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.__class__._global_last_request_time
            interval = self.__class__._global_request_interval
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
            self.__class__._global_last_request_time = asyncio.get_event_loop().time()

    @staticmethod
    def _clean(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _parse_entry(self, entry: ET.Element) -> RetrievedPaper:
        title = self._clean(entry.findtext("atom:title", default="", namespaces=_NS))
        abstract = self._clean(entry.findtext("atom:summary", default="", namespaces=_NS))

        authors: list[str] = []
        for author in entry.findall("atom:author", _NS):
            name = author.findtext("atom:name", default="", namespaces=_NS)
            if name:
                authors.append(name.strip())

        year: int | None = None
        published = entry.findtext("atom:published", default="", namespaces=_NS)
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                year = None

        # arxiv:journal_ref 优先, 否则 None
        venue = entry.findtext("arxiv:journal_ref", default=None, namespaces=_NS)
        venue = self._clean(venue) or None

        doi = entry.findtext("arxiv:doi", default=None, namespaces=_NS)
        doi = doi.strip() if doi else None

        # entry id 形如 http://arxiv.org/abs/2106.09685v1
        url = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip() or None

        return RetrievedPaper(
            source=self.source_name,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            abstract=abstract or None,
            url=url,
        )

    def _parse_feed(self, xml_text: str) -> list[RetrievedPaper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"[arXiv] XML parse error: {e}")
            return []
        return [self._parse_entry(e) for e in root.findall("atom:entry", _NS)]

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        # arXiv 字段化检索: ti:"<title>"
        safe = title.replace('"', "")
        params = {
            "search_query": f'ti:"{safe}"',
            "start": 0,
            "max_results": self.max_results,
        }
        resp = await self._request_with_retry("GET", self.base_url, params=params)
        papers = self._parse_feed(resp.text)
        logger.debug(f"[arXiv] title search → {len(papers)} results")
        return papers

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        # 取前两位作者: au:"X" AND au:"Y"
        au_parts = [f'au:"{a}"' for a in authors[:2] if a]
        if not au_parts:
            return []
        query = " AND ".join(au_parts)
        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.max_results,
        }
        resp = await self._request_with_retry("GET", self.base_url, params=params)
        papers = self._parse_feed(resp.text)
        # 客户端按年份过滤 (arXiv 检索语法对年份支持有限)
        if year:
            papers = [p for p in papers if p.year == year]
        return papers
