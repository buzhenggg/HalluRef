"""Google Scholar 检索器 — 直接 HTTP 爬取

作为 OpenAlex + CrossRef 的补查 (fallback)，仅在主力检索未找到时调用。
基于 httpx + BeautifulSoup 直接爬取 scholar.google.com，原生异步。
不依赖 scholarly 库（其内部 httpx 兼容性差、速度慢）。
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from src.models.schemas import RetrievedPaper


_BASE_URL = "https://scholar.google.com/scholar"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class GoogleScholarRetriever:
    """Google Scholar 检索器（直接 HTTP 爬取）

    Args:
        proxy: HTTP/SOCKS5 代理地址，如 "http://127.0.0.1:7890"。
               为 None 时不使用代理（容易被 Google 封 IP）。
        timeout: 单次检索超时秒数
        max_results: 每次检索返回的最大结果数
    """

    source_name = "google_scholar"

    def __init__(
        self,
        proxy: str | None = None,
        timeout: int = 15,
        max_results: int = 3,
    ):
        self.proxy = proxy
        self.timeout = timeout
        self.max_results = max_results
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs: dict = {
                "timeout": self.timeout,
                "headers": _DEFAULT_HEADERS,
                "follow_redirects": True,
            }
            if self.proxy:
                kwargs["proxy"] = self.proxy
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    @staticmethod
    def _detect_block(html: str) -> str | None:
        """检测是否被 Google 反爬挑战"""
        lower = html.lower()
        if "captcha" in lower:
            return "CAPTCHA challenge"
        if "unusual traffic" in lower:
            return "Unusual traffic block"
        if "sorry/index" in lower:
            return "Sorry page"
        if len(html) < 500:
            return f"Suspiciously short response ({len(html)} bytes)"
        return None

    @staticmethod
    def _parse_meta(meta: str) -> tuple[list[str], str | None, int | None]:
        """解析 'A Vaswani, N Shazeer - Advances in NeurIPS, 2017 - publisher' 行

        Returns: (authors, venue, year)
        """
        authors: list[str] = []
        venue: str | None = None
        year: int | None = None

        if not meta:
            return authors, venue, year

        # 归一化非常规空白和破折号 (nbsp / 零宽空格 / en-dash / em-dash)
        meta = meta.replace("\xa0", " ").replace("\u200b", "")
        # 按破折号分隔符切分: [authors_part, venue_year_part, publisher_part?]
        parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", meta)]

        # 作者部分
        if parts:
            authors_part = parts[0]
            # 兼容中英文逗号
            raw_authors = re.split(r"[,，]", authors_part)
            authors = [a.strip() for a in raw_authors if a.strip()]

        # venue + year 部分
        if len(parts) >= 2:
            venue_year = parts[1]
            # 提取末尾的 4 位年份
            year_match = re.search(r"(19|20)\d{2}", venue_year)
            if year_match:
                try:
                    year = int(year_match.group())
                except ValueError:
                    pass
                # 去掉年份及其前的逗号得到 venue
                venue = re.sub(r"[,，]?\s*(19|20)\d{2}.*$", "", venue_year).strip()
                # 去掉末尾省略号
                venue = venue.rstrip("…").rstrip(".").strip()
            else:
                venue = venue_year.rstrip("…").strip()

            if not venue:
                venue = None

        return authors, venue, year

    def _parse_html(self, html: str) -> list[RetrievedPaper]:
        """解析 Google Scholar 搜索结果页"""
        soup = BeautifulSoup(html, "html.parser")
        papers: list[RetrievedPaper] = []

        for div in soup.select("div.gs_ri"):
            title_tag = div.select_one("h3.gs_rt")
            if not title_tag:
                continue

            # 标题（去掉 [PDF] / [HTML] 等前缀标签）
            for label in title_tag.select("span.gs_ctg2"):
                label.decompose()
            title = title_tag.get_text(" ", strip=True)
            # 去掉残留的 [PDF] [HTML] [BOOK] 前缀
            title = re.sub(r"^(\[\w+\]\s*)+", "", title).strip()

            link = ""
            a = title_tag.find("a")
            if a:
                link = a.get("href", "")

            meta_tag = div.select_one("div.gs_a")
            meta = meta_tag.get_text(" ", strip=True) if meta_tag else ""
            authors, venue, year = self._parse_meta(meta)

            snippet_tag = div.select_one("div.gs_rs")
            abstract = snippet_tag.get_text(" ", strip=True) if snippet_tag else None

            papers.append(RetrievedPaper(
                source=self.source_name,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=None,
                abstract=abstract or None,
                url=link or None,
            ))

        return papers

    async def _fetch(self, query: str) -> list[RetrievedPaper]:
        """发起一次搜索请求"""
        if not query:
            return []
        try:
            client = await self._get_client()
            resp = await client.get(_BASE_URL, params={"q": query, "hl": "en"})
        except Exception as e:
            logger.warning(f"[GoogleScholar] request error: {e}")
            return []

        if resp.status_code != 200:
            logger.warning(f"[GoogleScholar] HTTP {resp.status_code}")
            return []

        block = self._detect_block(resp.text)
        if block:
            logger.warning(f"[GoogleScholar] blocked: {block}")
            return []

        try:
            papers = self._parse_html(resp.text)
        except Exception as e:
            logger.warning(f"[GoogleScholar] parse error: {e}")
            return []

        return papers[: self.max_results]

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        return await self._fetch(title)

    async def search_by_author_year(
        self, authors: list[str], year: int | None
    ) -> list[RetrievedPaper]:
        query = " ".join(authors[:2])
        if year:
            query += f" {year}"
        return await self._fetch(query)

    async def search(
        self,
        title: str = "",
        authors: list[str] | None = None,
        year: int | None = None,
    ) -> list[RetrievedPaper]:
        """综合检索：先按标题，无结果则按作者+年份"""
        results: list[RetrievedPaper] = []
        if title:
            results = await self.search_by_title(title)
        if not results and authors:
            results = await self.search_by_author_year(authors, year)
        return results

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def is_configured(self) -> bool:
        """无强制配置项 (proxy 可选)。"""
        return True
