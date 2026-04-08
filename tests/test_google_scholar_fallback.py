"""Google Scholar 补查（Fallback）测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.models.schemas import (
    Citation,
    ParsedCitation,
    RetrievedPaper,
)
from src.retrievers.base import BaseRetriever
from src.retrievers.google_scholar import GoogleScholarRetriever
from src.agents.reference_retriever import ReferenceRetriever


# ── 辅助工厂 ──────────────────────────────────────────────


def _make_citation(
    cid: str, title: str, authors: list[str] | None = None, year: int | None = None
) -> Citation:
    return Citation(
        citation_id=cid,
        raw_text=f"[{cid}]",
        parsed=ParsedCitation(title=title, authors=authors or [], year=year),
    )


def _make_paper(source: str, title: str) -> RetrievedPaper:
    return RetrievedPaper(source=source, title=title)


class FakeRetriever(BaseRetriever):
    """总是返回空结果的检索器，模拟主力检索失败"""

    def __init__(self, name: str, papers: list[RetrievedPaper] | None = None):
        super().__init__(timeout=5, request_interval=0.0, max_retries=0)
        self.source_name = name
        self._papers = papers or []

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        return self._papers

    async def search_by_author_year(self, authors: list[str], year: int | None) -> list[RetrievedPaper]:
        return self._papers


# ── 模拟的 Google Scholar HTML 响应 ───────────────────────


_FAKE_HTML = """
<html><body>
<div class="gs_ri">
  <h3 class="gs_rt"><a href="https://example.com/paper1">Attention Is All You Need</a></h3>
  <div class="gs_a">A Vaswani, N Shazeer, N Parmar - Advances in NeurIPS, 2017 - proceedings.neurips.cc</div>
  <div class="gs_rs">The dominant sequence transduction models are based on complex recurrent...</div>
  <div class="gs_fl"><a>Cited by 240065</a></div>
</div>
<div class="gs_ri">
  <h3 class="gs_rt"><a href="https://example.com/paper2">Another Paper</a></h3>
  <div class="gs_a">B Author - Some Venue, 2020 - example.com</div>
  <div class="gs_rs">Brief abstract snippet...</div>
</div>
</body></html>
"""

_CHINESE_HTML = """
<html><body>
<div class="gs_ri">
  <h3 class="gs_rt"><a href="https://example.com/zh">知识图谱构建技术综述</a></h3>
  <div class="gs_a">刘峤, 李杨, 段宏, 刘瑶, 秦志光 - 计算机研究与发展, 2016 - cdn.jsdelivr.net</div>
  <div class="gs_rs">本文从知识图谱的构建角度出发,深度剖析知识图谱概念...</div>
  <div class="gs_fl"><a>被引用次数: 270</a></div>
</div>
</body></html>
"""

_BLOCKED_HTML = "<html><body>Our systems have detected unusual traffic from your computer network.</body></html>"


def _mock_response(html: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=html,
        request=httpx.Request("GET", "https://scholar.google.com/scholar"),
    )


# ── GoogleScholarRetriever 单元测试 ───────────────────────


class TestParseHtml:
    """测试 HTML 解析逻辑"""

    def test_parse_basic_results(self):
        retriever = GoogleScholarRetriever(proxy=None)
        papers = retriever._parse_html(_FAKE_HTML)

        assert len(papers) == 2

        p1 = papers[0]
        assert p1.source == "google_scholar"
        assert p1.title == "Attention Is All You Need"
        assert "Vaswani" in p1.authors[0]
        assert p1.year == 2017
        assert p1.venue is not None and "NeurIPS" in p1.venue
        assert p1.abstract and "transduction" in p1.abstract
        assert p1.url == "https://example.com/paper1"

    def test_parse_chinese_paper(self):
        retriever = GoogleScholarRetriever(proxy=None)
        papers = retriever._parse_html(_CHINESE_HTML)

        assert len(papers) == 1
        p = papers[0]
        assert p.title == "知识图谱构建技术综述"
        assert any("刘峤" in a for a in p.authors)
        assert p.year == 2016
        assert p.venue is not None and "计算机研究与发展" in p.venue

    def test_parse_empty_html(self):
        retriever = GoogleScholarRetriever(proxy=None)
        papers = retriever._parse_html("<html><body></body></html>")
        assert papers == []

    def test_detect_block_unusual_traffic(self):
        retriever = GoogleScholarRetriever(proxy=None)
        assert retriever._detect_block(_BLOCKED_HTML) is not None

    def test_detect_block_normal_html(self):
        retriever = GoogleScholarRetriever(proxy=None)
        assert retriever._detect_block(_FAKE_HTML) is None


class TestSearch:
    """测试 search 方法（mock httpx）"""

    @pytest.mark.asyncio
    async def test_search_by_title_success(self):
        retriever = GoogleScholarRetriever(proxy=None, max_results=2)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=_mock_response(_FAKE_HTML))
        retriever._client = mock_client

        papers = await retriever.search_by_title("Attention")

        assert len(papers) == 2
        assert papers[0].title == "Attention Is All You Need"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_max_results_truncates(self):
        retriever = GoogleScholarRetriever(proxy=None, max_results=1)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=_mock_response(_FAKE_HTML))
        retriever._client = mock_client

        papers = await retriever.search_by_title("test")
        assert len(papers) == 1

    @pytest.mark.asyncio
    async def test_search_blocked_returns_empty(self):
        retriever = GoogleScholarRetriever(proxy=None)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=_mock_response(_BLOCKED_HTML))
        retriever._client = mock_client

        papers = await retriever.search_by_title("test")
        assert papers == []

    @pytest.mark.asyncio
    async def test_search_http_error_returns_empty(self):
        retriever = GoogleScholarRetriever(proxy=None)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("network error"))
        retriever._client = mock_client

        papers = await retriever.search_by_title("test")
        assert papers == []

    @pytest.mark.asyncio
    async def test_search_falls_back_to_author(self):
        """无标题但有作者时应按作者检索"""
        retriever = GoogleScholarRetriever(proxy=None)

        mock_client = MagicMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=_mock_response(_FAKE_HTML))
        retriever._client = mock_client

        papers = await retriever.search(title="", authors=["Vaswani"], year=2017)
        assert len(papers) > 0


# ── ReferenceRetriever fallback 集成测试 ──────────────────


class TestFallbackIntegration:
    """测试主力检索失败后 Google Scholar 补查"""

    @pytest.mark.asyncio
    async def test_fallback_triggered_when_primary_not_found(self):
        primary = FakeRetriever("openalex", papers=[])
        fallback = GoogleScholarRetriever(proxy=None)
        gs_paper = RetrievedPaper(source="google_scholar", title="Attention Is All You Need", year=2017)
        fallback.search = AsyncMock(return_value=[gs_paper])

        agent = ReferenceRetriever(retrievers=[primary], fallback_retriever=fallback)
        result = await agent.verify(_make_citation("ref_001", "Attention Is All You Need"))

        assert result.found is True
        assert result.best_match.source == "google_scholar"
        fallback.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_not_triggered_when_primary_found(self):
        oa_paper = _make_paper("openalex", "Attention Is All You Need")
        primary = FakeRetriever("openalex", papers=[oa_paper])
        fallback = GoogleScholarRetriever(proxy=None)
        fallback.search = AsyncMock(return_value=[])

        agent = ReferenceRetriever(retrievers=[primary], fallback_retriever=fallback)
        result = await agent.verify(_make_citation("ref_001", "Attention Is All You Need"))

        assert result.found is True
        assert result.best_match.source == "openalex"
        fallback.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_failure_does_not_crash(self):
        primary = FakeRetriever("openalex", papers=[])
        fallback = GoogleScholarRetriever(proxy=None)
        fallback.search = AsyncMock(side_effect=Exception("Google blocked"))

        agent = ReferenceRetriever(retrievers=[primary], fallback_retriever=fallback)
        result = await agent.verify(_make_citation("ref_001", "Some Paper"))

        assert result.found is False

    @pytest.mark.asyncio
    async def test_no_fallback_configured(self):
        primary = FakeRetriever("openalex", papers=[])
        agent = ReferenceRetriever(retrievers=[primary], fallback_retriever=None)
        result = await agent.verify(_make_citation("ref_001", "Nonexistent"))
        assert result.found is False

    @pytest.mark.asyncio
    async def test_fallback_in_batch(self):
        primary = FakeRetriever("openalex", papers=[])
        gs_paper = _make_paper("google_scholar", "Paper A")
        fallback = GoogleScholarRetriever(proxy=None)
        fallback.search = AsyncMock(return_value=[gs_paper])

        agent = ReferenceRetriever(retrievers=[primary], fallback_retriever=fallback)
        results = await agent.verify_batch([
            _make_citation("ref_001", "Paper A"),
            _make_citation("ref_002", "Paper A"),
        ])
        assert len(results) == 2
        assert all(r.found for r in results)
