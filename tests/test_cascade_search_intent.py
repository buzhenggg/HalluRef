"""CascadeRetriever tests for academic + scholar API + direct Scholar fallback."""

from __future__ import annotations

import asyncio

import pytest

from src.models.schemas import MatchConfidence, RetrievedPaper
from src.retrievers.base import BaseRetriever
from src.retrievers.cascade import CascadeRetriever


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


def _paper(source: str, title: str) -> RetrievedPaper:
    return RetrievedPaper(source=source, title=title)


class TestEarlyStop:
    @pytest.mark.asyncio
    async def test_primary_academic_hit_skips_secondary_and_search_layers(self):
        arxiv = FakeRetriever("arxiv", [_paper("arxiv", "Attention Is All You Need")])
        semantic = FakeRetriever("semantic_scholar", [_paper("semantic_scholar", "Attention Is All You Need")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [_paper("openalex", "Attention Is All You Need")]),
            crossref=FakeRetriever("crossref", [_paper("crossref", "Attention Is All You Need")]),
            arxiv=arxiv,
            semantic_scholar=semantic,
            scholar_search=FakeRetriever("serper_scholar", [_paper("serper_scholar", "X")]),
            google_scholar_direct=FakeRetriever("google_scholar", [_paper("google_scholar", "Z")]),
        )
        res = await cas.search(title="Attention Is All You Need")

        assert res.found is True
        assert res.confidence == MatchConfidence.HIGH
        assert res.hit_tier == "academic_primary"
        assert res.tiers_run == ["academic_primary"]
        assert arxiv.call_count == 0
        assert semantic.call_count == 0


class TestCascadeFallback:
    def test_scholar_api_configuration_disables_direct_scholar_tier(self):
        direct = FakeRetriever("google_scholar", [_paper("google_scholar", "Target Title")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", []),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            scholar_search=FakeRetriever("serper_scholar", [_paper("serper_scholar", "Target Title")]),
            google_scholar_direct=direct,
        )
        assert [name for name, _, _ in cas.tiers] == [
            "academic_primary",
            "academic_secondary",
            "scholar_search",
        ]

    @pytest.mark.asyncio
    async def test_scholar_api_runs_without_direct_scholar(self):
        direct = FakeRetriever("google_scholar", [_paper("google_scholar", "Target Title")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", []),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            scholar_search=FakeRetriever("serper_scholar", [_paper("serper_scholar", "Target Title")]),
            google_scholar_direct=direct,
        )
        res = await cas.search(title="Target Title")

        assert res.found is True
        assert res.hit_tier == "scholar_search"
        assert res.best_match.source == "serper_scholar"
        assert direct.call_count == 0
        assert res.tiers_run == ["academic_primary", "academic_secondary", "scholar_search"]

    @pytest.mark.asyncio
    async def test_direct_scholar_does_not_run_after_configured_scholar_api_miss(self):
        scholar_api = FakeRetriever("serper_scholar", [])
        direct = FakeRetriever("google_scholar", [_paper("google_scholar", "Target Title")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", []),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            scholar_search=scholar_api,
            google_scholar_direct=direct,
        )
        res = await cas.search(title="Target Title")

        assert res.found is False
        assert res.hit_tier is None
        assert scholar_api.call_count == 1
        assert direct.call_count == 0
        assert res.tiers_run == ["academic_primary", "academic_secondary", "scholar_search"]

    @pytest.mark.asyncio
    async def test_direct_scholar_used_when_no_scholar_api(self):
        direct = FakeRetriever("google_scholar", [_paper("google_scholar", "Target Title")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", []),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            google_scholar_direct=direct,
        )
        res = await cas.search(title="Target Title")

        assert res.found is True
        assert res.hit_tier == "google_scholar_direct"
        assert res.best_match.source == "google_scholar"
        assert direct.call_count == 1

    @pytest.mark.asyncio
    async def test_all_miss_accumulates_candidates(self):
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [_paper("openalex", "Totally Different")]),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            scholar_search=FakeRetriever("serper_scholar", []),
            google_scholar_direct=FakeRetriever("google_scholar", []),
        )
        res = await cas.search(title="Target Title")

        assert res.found is False
        assert res.tiers_run == ["academic_primary", "academic_secondary", "scholar_search"]
        assert any(p.source == "openalex" for p in res.candidates)


class TestSkipUnconfigured:
    @pytest.mark.asyncio
    async def test_unconfigured_scholar_api_skips_to_direct_scholar(self):
        scholar_api = FakeRetriever("serper_scholar", [], configured=False)
        direct = FakeRetriever("google_scholar", [_paper("google_scholar", "Target")])
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", []),
            crossref=FakeRetriever("crossref", []),
            arxiv=FakeRetriever("arxiv", []),
            semantic_scholar=FakeRetriever("semantic_scholar", []),
            scholar_search=scholar_api,
            google_scholar_direct=direct,
        )
        res = await cas.search(title="Target")

        assert res.hit_tier == "google_scholar_direct"
        assert scholar_api.call_count == 0
        assert direct.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_scholar_api_preferred_over_fallback(self):
        primary = FakeRetriever("serper_scholar", [_paper("serper_scholar", "Target")])
        fallback = FakeRetriever("serpapi_scholar", [_paper("serpapi_scholar", "Target")])
        cas = CascadeRetriever(scholar_search=primary, scholar_search_fallback=fallback)
        res = await cas.search(title="Target")
        assert res.best_match.source == "serper_scholar"
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_scholar_api_used_when_primary_missing(self):
        primary = FakeRetriever("serper_scholar", [], configured=False)
        fallback = FakeRetriever("serpapi_scholar", [_paper("serpapi_scholar", "Target")])
        cas = CascadeRetriever(scholar_search=primary, scholar_search_fallback=fallback)
        res = await cas.search(title="Target")
        assert res.best_match.source == "serpapi_scholar"
        assert primary.call_count == 0


class TestResilience:
    @pytest.mark.asyncio
    async def test_one_failing_retriever_in_academic_tier_doesnt_block(self):
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [], should_fail=True),
            crossref=FakeRetriever("crossref", [_paper("crossref", "Target")]),
        )
        res = await cas.search(title="Target")
        assert res.found is True
        assert res.best_match.source == "crossref"

    @pytest.mark.asyncio
    async def test_falling_through_after_academic_failure(self):
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [], should_fail=True),
            crossref=FakeRetriever("crossref", [], should_fail=True),
            arxiv=FakeRetriever("arxiv", [], should_fail=True),
            semantic_scholar=FakeRetriever("semantic_scholar", [], should_fail=True),
            scholar_search=FakeRetriever("serper_scholar", [_paper("serper_scholar", "Target")]),
        )
        res = await cas.search(title="Target")
        assert res.hit_tier == "scholar_search"


class TestParallelTier1:
    @pytest.mark.asyncio
    async def test_primary_academic_runs_in_parallel(self):
        delay = 0.2
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [_paper("openalex", "T")], delay=delay),
            crossref=FakeRetriever("crossref", [_paper("crossref", "T")], delay=delay),
            arxiv=FakeRetriever("arxiv", [_paper("arxiv", "T")], delay=delay),
            semantic_scholar=FakeRetriever("semantic_scholar", [_paper("semantic_scholar", "T")], delay=delay),
        )
        start = asyncio.get_event_loop().time()
        await cas.search(title="T")
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.35

    @pytest.mark.asyncio
    async def test_secondary_academic_runs_only_after_primary_miss(self):
        openalex = FakeRetriever("openalex", [], delay=0.2)
        crossref = FakeRetriever("crossref", [], delay=0.2)
        arxiv = FakeRetriever("arxiv", [_paper("arxiv", "T")], delay=0.2)
        semantic = FakeRetriever("semantic_scholar", [_paper("semantic_scholar", "T")], delay=0.2)
        cas = CascadeRetriever(
            openalex=openalex,
            crossref=crossref,
            arxiv=arxiv,
            semantic_scholar=semantic,
        )
        start = asyncio.get_event_loop().time()
        res = await cas.search(title="T")
        elapsed = asyncio.get_event_loop().time() - start

        assert res.hit_tier == "academic_secondary"
        assert res.tiers_run == ["academic_primary", "academic_secondary"]
        assert elapsed >= 0.35
        assert elapsed < 0.55


class TestSourcePriority:
    @pytest.mark.asyncio
    async def test_openalex_wins_on_tie(self):
        cas = CascadeRetriever(
            openalex=FakeRetriever("openalex", [_paper("openalex", "Same Title")]),
            crossref=FakeRetriever("crossref", [_paper("crossref", "Same Title")]),
        )
        res = await cas.search(title="Same Title")
        assert res.best_match.source == "openalex"

    @pytest.mark.asyncio
    async def test_arxiv_beats_scholar_api_on_tie(self):
        cas = CascadeRetriever(
            arxiv=FakeRetriever("arxiv", [_paper("arxiv", "Same Title")]),
            scholar_search=FakeRetriever("serper_scholar", [_paper("serper_scholar", "Same Title")]),
        )
        res = await cas.search(title="Same Title")
        assert res.best_match.source == "arxiv"
        assert res.hit_tier == "academic_secondary"

    @pytest.mark.asyncio
    async def test_semantic_scholar_beats_search_api_on_tie(self):
        cas = CascadeRetriever(
            semantic_scholar=FakeRetriever(
                "semantic_scholar", [_paper("semantic_scholar", "Same Title")]
            ),
            scholar_search=FakeRetriever(
                "serper_scholar", [_paper("serper_scholar", "Same Title")]
            ),
        )
        res = await cas.search(title="Same Title")
        assert res.best_match.source == "semantic_scholar"
        assert res.hit_tier == "academic_secondary"
