"""Retrieval debug log tests."""

from __future__ import annotations

import pytest

from src.retrievers.cascade import CascadeRetriever
from src.models.schemas import RetrievedPaper
from src.retrievers.base import BaseRetriever


class FakeRetriever(BaseRetriever):
    def __init__(self, name: str, papers=None, should_fail: bool = False):
        super().__init__(timeout=5, request_interval=0.0, max_retries=0)
        self.source_name = name
        self._papers = papers or []
        self._should_fail = should_fail

    def is_configured(self) -> bool:
        return True

    async def search_by_title(self, title):
        if self._should_fail:
            raise RuntimeError("boom")
        return list(self._papers)

    async def search_by_author_year(self, authors, year):
        return await self.search_by_title("")


@pytest.mark.asyncio
async def test_debug_log_records_failures_and_misses():
    cascade = CascadeRetriever(
        openalex=FakeRetriever("openalex", should_fail=True),
        crossref=FakeRetriever("crossref", []),
        arxiv=FakeRetriever("arxiv", []),
        scholar_search=FakeRetriever("serper_scholar", []),
    )

    result = await cascade.search(title="Missing Paper")

    assert result.found is False
    assert "academic_primary/openalex" in result.debug_log
    assert "error" in result.debug_log.lower()
    assert "academic_primary/crossref" in result.debug_log
    assert "academic_secondary/arxiv" in result.debug_log
    assert "0 candidates" in result.debug_log
    assert "scholar_search/serper_scholar" in result.debug_log


@pytest.mark.asyncio
async def test_debug_log_records_hit_tier_summary():
    cascade = CascadeRetriever(
        scholar_search=FakeRetriever(
            "serper_scholar",
            [RetrievedPaper(source="serper_scholar", title="Found Paper")],
        ),
    )

    result = await cascade.search(title="Found Paper")

    assert result.found is True
    assert "最终命中层: scholar_search" in result.debug_log
