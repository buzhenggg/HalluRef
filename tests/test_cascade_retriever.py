"""CascadeRetriever 级联编排测试"""

from __future__ import annotations

import asyncio

import pytest

from src.models.schemas import MatchConfidence, RetrievedPaper
from src.retrievers.base import BaseRetriever
from src.retrievers.cascade import CascadeRetriever


# ── Fakes ─────────────────────────────────────────────────


class FakeRetriever(BaseRetriever):
    def __init__(
        self,
        name: str,
        papers: list[RetrievedPaper] | None = None,
        configured: bool = True,
        delay: float = 0.0,
        should_fail: bool = False,
    ):
        super().__init__(timeout=5, request_interval=0.0, max_retries=0)
        self.source_name = name
        self._papers = papers or []
        self._configured = configured
        self._delay = delay
        self._should_fail = should_fail
        self.call_count = 0

    def is_configured(self) -> bool:
        return self._configured

    async def search_by_title(self, title):
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"{self.source_name} fail")
        return list(self._papers)

    async def search_by_author_year(self, authors, year):
        return await self.search_by_title("")


class FakeScholar:
    """模拟 GoogleScholarRetriever (不继承 BaseRetriever)"""

    def __init__(self, papers=None, configured=True, should_fail=False):
        self.source_name = "google_scholar"
        self._papers = papers or []
        self._configured = configured
        self._fail = should_fail
        self.call_count = 0

    def is_configured(self):
        return self._configured

    async def search(self, title="", authors=None, year=None):
        self.call_count += 1
        if self._fail:
            raise RuntimeError("scholar fail")
        return list(self._papers)

    async def close(self):
        pass


def _paper(source: str, title: str) -> RetrievedPaper:
    return RetrievedPaper(source=source, title=title)


# ── 1. Tier 1 命中 → 后续 tier 不调用 ────────────────────


class TestEarlyStop:
    @pytest.mark.asyncio
    async def test_tier1_hit_skips_serper_and_scholar(self):
        oa = FakeRetriever("openalex", [_paper("openalex", "Attention Is All You Need")])
        cr = FakeRetriever("crossref", [_paper("crossref", "Attention Is All You Need")])
        sp = FakeRetriever("serper", [_paper("serper", "X")])
        gs = FakeScholar([_paper("google_scholar", "Y")])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp, scholar=gs)
        res = await cas.search(title="Attention Is All You Need")

        assert res.found is True
        assert res.confidence == MatchConfidence.HIGH
        assert res.hit_tier == "academic"
        assert sp.call_count == 0
        assert gs.call_count == 0
        # Tier 1 两个都被调用
        assert oa.call_count == 1
        assert cr.call_count == 1
        assert res.tiers_run == ["academic"]


# ── 2. Tier 1 未命中 → 顺次降级到 serper / scholar ─────


class TestCascadeFallback:
    @pytest.mark.asyncio
    async def test_falls_through_to_serper(self):
        oa = FakeRetriever("openalex", [])
        cr = FakeRetriever("crossref", [])
        sp = FakeRetriever("serper", [_paper("serper", "Target Title")])
        gs = FakeScholar([_paper("google_scholar", "Other")])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp, scholar=gs)
        res = await cas.search(title="Target Title")

        assert res.found is True
        assert res.hit_tier == "web_search"
        assert res.best_match.source == "serper"
        assert sp.call_count == 1
        assert gs.call_count == 0  # 已命中, scholar 不调用
        assert res.tiers_run == ["academic", "web_search"]

    @pytest.mark.asyncio
    async def test_falls_through_to_scholar(self):
        oa = FakeRetriever("openalex", [])
        cr = FakeRetriever("crossref", [])
        sp = FakeRetriever("serper", [])
        gs = FakeScholar([_paper("google_scholar", "Target Title")])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp, scholar=gs)
        res = await cas.search(title="Target Title")

        assert res.found is True
        assert res.hit_tier == "scholar"
        assert res.best_match.source == "google_scholar"
        assert res.tiers_run == ["academic", "web_search", "scholar"]

    @pytest.mark.asyncio
    async def test_all_miss(self):
        oa = FakeRetriever("openalex", [_paper("openalex", "Totally Different")])
        cr = FakeRetriever("crossref", [])
        sp = FakeRetriever("serper", [])
        gs = FakeScholar([])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp, scholar=gs)
        res = await cas.search(title="Target Title")

        assert res.found is False
        assert len(res.tiers_run) == 3
        # 候选累积: 至少 openalex 那条
        assert any(p.source == "openalex" for p in res.candidates)


# ── 3. 未配置的 tier 跳过 ─────────────────────────────


class TestSkipUnconfigured:
    @pytest.mark.asyncio
    async def test_serper_unconfigured_skipped(self):
        oa = FakeRetriever("openalex", [])
        cr = FakeRetriever("crossref", [])
        sp = FakeRetriever("serper", [_paper("serper", "X")], configured=False)
        gs = FakeScholar([_paper("google_scholar", "Target")])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp, scholar=gs)
        res = await cas.search(title="Target")

        assert sp.call_count == 0  # 未配置不调用
        assert res.hit_tier == "scholar"
        assert "web_search" not in res.tiers_run

    @pytest.mark.asyncio
    async def test_only_openalex_configured(self):
        oa = FakeRetriever("openalex", [_paper("openalex", "Target")])
        cas = CascadeRetriever(openalex=oa)
        res = await cas.search(title="Target")

        assert res.found is True
        assert res.tiers_run == ["academic"]

    @pytest.mark.asyncio
    async def test_no_retrievers_configured(self):
        cas = CascadeRetriever()
        res = await cas.search(title="X")
        assert res.found is False
        assert res.tiers_run == []

    @pytest.mark.asyncio
    async def test_tier2_serper_preferred_over_serpapi(self):
        """Tier 2 同时配 serper + serpapi 时, 仅启用 Serper"""
        sp = FakeRetriever("serper", [_paper("serper", "Target")])
        sa = FakeRetriever("serpapi", [_paper("serpapi", "Target")])
        cas = CascadeRetriever(serper=sp, serpapi=sa)
        res = await cas.search(title="Target")
        assert res.found is True
        assert res.best_match.source == "serper"
        assert sa.call_count == 0  # SerpAPI 完全不调用

    @pytest.mark.asyncio
    async def test_tier2_falls_back_to_serpapi_when_serper_unconfigured(self):
        sp = FakeRetriever("serper", [], configured=False)
        sa = FakeRetriever("serpapi", [_paper("serpapi", "Target")])
        cas = CascadeRetriever(serper=sp, serpapi=sa)
        res = await cas.search(title="Target")
        assert res.found is True
        assert res.best_match.source == "serpapi"
        assert sp.call_count == 0


# ── 4. 失败鲁棒性 ──────────────────────────────────────


class TestResilience:
    @pytest.mark.asyncio
    async def test_one_failing_retriever_in_tier_doesnt_block(self):
        oa = FakeRetriever("openalex", [], should_fail=True)
        cr = FakeRetriever("crossref", [_paper("crossref", "Target")])

        cas = CascadeRetriever(openalex=oa, crossref=cr)
        res = await cas.search(title="Target")

        assert res.found is True
        assert res.best_match.source == "crossref"

    @pytest.mark.asyncio
    async def test_tier_failure_falls_through(self):
        oa = FakeRetriever("openalex", [], should_fail=True)
        cr = FakeRetriever("crossref", [], should_fail=True)
        sp = FakeRetriever("serper", [_paper("serper", "Target")])

        cas = CascadeRetriever(openalex=oa, crossref=cr, serper=sp)
        res = await cas.search(title="Target")

        assert res.found is True
        assert res.hit_tier == "web_search"


# ── 5. Tier 1 并行 (而非串行) ─────────────────────────


class TestParallelTier1:
    @pytest.mark.asyncio
    async def test_tier1_runs_in_parallel(self):
        delay = 0.2
        oa = FakeRetriever("openalex", [_paper("openalex", "T")], delay=delay)
        cr = FakeRetriever("crossref", [_paper("crossref", "T")], delay=delay)

        cas = CascadeRetriever(openalex=oa, crossref=cr)
        start = asyncio.get_event_loop().time()
        await cas.search(title="T")
        elapsed = asyncio.get_event_loop().time() - start
        # 并行 ~0.2s, 串行 ~0.4s
        assert elapsed < 0.35


# ── 6. 来源优先级 tiebreak: openalex > crossref ────────


class TestSourcePriority:
    @pytest.mark.asyncio
    async def test_openalex_wins_on_tie(self):
        # 两条标题完全相同 → score 相同 → openalex 优先
        oa = FakeRetriever("openalex", [_paper("openalex", "Same Title")])
        cr = FakeRetriever("crossref", [_paper("crossref", "Same Title")])

        cas = CascadeRetriever(openalex=oa, crossref=cr)
        res = await cas.search(title="Same Title")

        assert res.found is True
        assert res.best_match.source == "openalex"
