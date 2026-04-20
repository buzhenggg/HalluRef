"""ReferenceRetriever 并发检索测试（OpenAlex + CrossRef）"""

import asyncio

import pytest

from src.models.schemas import (
    Citation,
    MatchConfidence,
    ParsedCitation,
    RetrievedPaper,
)
from src.retrievers.base import BaseRetriever
from src.agents.reference_retriever import ReferenceRetriever


# ── 模拟检索器 ────────────────────────────────────────────


class FakeRetriever(BaseRetriever):
    """可控的假检索器，用于测试并发逻辑"""

    def __init__(self, name: str, papers: list[RetrievedPaper], delay: float = 0.0, should_fail: bool = False):
        super().__init__(timeout=5, request_interval=0.0, max_retries=0)
        self.source_name = name
        self._papers = papers
        self._delay = delay
        self._should_fail = should_fail
        self.call_count = 0

    async def search_by_title(self, title: str) -> list[RetrievedPaper]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"{self.source_name} simulated failure")
        return self._papers

    async def search_by_author_year(self, authors: list[str], year: int | None) -> list[RetrievedPaper]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"{self.source_name} simulated failure")
        return self._papers


# ── 辅助工厂 ──────────────────────────────────────────────


def _make_citation(cid: str, title: str, authors: list[str] | None = None, year: int | None = None) -> Citation:
    return Citation(
        citation_id=cid,
        raw_text=f"[{cid}]",
        parsed=ParsedCitation(title=title, authors=authors or [], year=year),
    )


def _make_paper(source: str, title: str, authors: list[str] | None = None, year: int | None = None) -> RetrievedPaper:
    return RetrievedPaper(source=source, title=title, authors=authors or [], year=year)


# ── 测试: 并发调用所有检索器 ──────────────────────────────


class TestConcurrentVerify:
    """测试 verify 方法并发调用所有检索器"""

    @pytest.mark.asyncio
    async def test_all_retrievers_called_concurrently(self):
        """所有检索器都应被调用（而非串行提前终止）"""
        r1 = FakeRetriever("openalex", [_make_paper("openalex", "Attention Is All You Need")])
        r2 = FakeRetriever("crossref", [_make_paper("crossref", "Attention Is All You Need")])

        agent = ReferenceRetriever(retrievers=[r1, r2], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "Attention Is All You Need")

        result = await agent.verify(citation)

        # 所有检索器都应被调用
        assert r1.call_count >= 1
        assert r2.call_count >= 1
        # 应找到匹配
        assert result.found is True
        assert result.confidence == MatchConfidence.HIGH

    @pytest.mark.asyncio
    async def test_candidates_merged_from_all_sources(self):
        """所有检索器的结果应被合并"""
        r1 = FakeRetriever("openalex", [_make_paper("openalex", "Paper A")])
        r2 = FakeRetriever("crossref", [_make_paper("crossref", "Paper B")])

        agent = ReferenceRetriever(retrievers=[r1, r2], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "Paper A")

        result = await agent.verify(citation)

        sources = {p.source for p in result.all_candidates}
        assert "openalex" in sources
        assert "crossref" in sources

    @pytest.mark.asyncio
    async def test_single_retriever_failure_does_not_block(self):
        """单个检索器失败不影响其他检索器返回结果"""
        r_fail = FakeRetriever("openalex", [], should_fail=True)
        r_ok = FakeRetriever("crossref", [_make_paper("crossref", "Attention Is All You Need")])

        agent = ReferenceRetriever(retrievers=[r_fail, r_ok], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "Attention Is All You Need")

        result = await agent.verify(citation)

        assert result.found is True
        assert result.best_match is not None
        assert result.best_match.source == "crossref"

    @pytest.mark.asyncio
    async def test_all_retrievers_fail(self):
        """所有检索器都失败时应返回未找到"""
        r1 = FakeRetriever("openalex", [], should_fail=True)
        r2 = FakeRetriever("crossref", [], should_fail=True)

        agent = ReferenceRetriever(retrievers=[r1, r2], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "Some Paper")

        result = await agent.verify(citation)

        assert result.found is False
        assert result.confidence == MatchConfidence.NONE

    @pytest.mark.asyncio
    async def test_concurrency_is_faster_than_serial(self):
        """并发应比串行快：2个各耗时0.2s的检索器，总时间应远小于0.4s"""
        delay = 0.2
        paper = _make_paper("src", "Test Paper")
        r1 = FakeRetriever("openalex", [paper], delay=delay)
        r2 = FakeRetriever("crossref", [paper], delay=delay)

        agent = ReferenceRetriever(retrievers=[r1, r2], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "Test Paper")

        start = asyncio.get_event_loop().time()
        await agent.verify(citation)
        elapsed = asyncio.get_event_loop().time() - start

        # 并发执行应在 ~0.2s 完成，远小于串行的 0.4s
        assert elapsed < 0.35


# ── 测试: 批量并发 ───────────────────────────────────────


class TestConcurrentVerifyBatch:
    """测试 verify_batch 方法的并发控制"""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_order(self):
        """批量结果顺序应与输入一致"""
        r = FakeRetriever("openalex", [_make_paper("openalex", "Paper")])
        agent = ReferenceRetriever(retrievers=[r], interval_min=0.0, interval_max=0.0)

        citations = [_make_citation(f"ref_{i:03d}", "Paper") for i in range(5)]
        results = await agent.verify_batch(citations)

        assert len(results) == 5
        for i, res in enumerate(results):
            assert res.citation_id == f"ref_{i:03d}"

    @pytest.mark.asyncio
    async def test_batch_concurrency_limited(self):
        """批量验证当前会被全局入口锁串行化"""
        paper = _make_paper("openalex", "Paper")
        r = FakeRetriever("openalex", [paper], delay=0.1)
        agent = ReferenceRetriever(
            retrievers=[r], max_concurrent=2, interval_min=0.0, interval_max=0.0
        )

        citations = [_make_citation(f"ref_{i:03d}", "Paper") for i in range(4)]

        start = asyncio.get_event_loop().time()
        results = await agent.verify_batch(citations)
        elapsed = asyncio.get_event_loop().time() - start

        assert len(results) == 4
        # 当前实现使用全局入口锁串行化 verify，因此总耗时接近 4 * 0.1s
        assert elapsed >= 0.4

    @pytest.mark.asyncio
    async def test_batch_empty(self):
        """空列表应返回空结果"""
        r = FakeRetriever("openalex", [])
        agent = ReferenceRetriever(retrievers=[r], interval_min=0.0, interval_max=0.0)

        results = await agent.verify_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_no_title_no_authors(self):
        """无标题无作者时应返回 NONE 置信度"""
        r = FakeRetriever("openalex", [])
        agent = ReferenceRetriever(retrievers=[r], interval_min=0.0, interval_max=0.0)
        citation = _make_citation("ref_001", "", authors=[])

        result = await agent.verify(citation)
        assert result.found is False
        assert result.confidence == MatchConfidence.NONE


def test_default_random_interval_range_is_one_to_three_seconds():
    agent = ReferenceRetriever()

    assert agent.interval_min == 1.0
    assert agent.interval_max == 3.0
