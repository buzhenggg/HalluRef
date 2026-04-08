"""Agent 1: CitationExtractor 单元测试"""

from __future__ import annotations

import pytest

from src.agents.citation_extractor import CitationExtractor


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def chat_json(self, prompt, system=None):
        self.calls += 1
        return self.payload


@pytest.mark.asyncio
async def test_skip_citation_without_title():
    payload = [
        {
            "authors": ["Alice"],
            "title": "Real Paper",
            "year": 2023,
            "context": "as shown in [1]",
            "claim": "X improves Y",
        },
        {
            "authors": ["Bob"],
            "title": "",
            "year": 2022,
            "context": "see [2]",
            "claim": "Z holds",
        },
        {
            "authors": ["Carol"],
            "title": "   ",
            "year": 2021,
            "context": "see [3]",
            "claim": "W",
        },
    ]
    extractor = CitationExtractor(_FakeLLM(payload))
    citations = await extractor.extract("dummy text")
    assert len(citations) == 1
    assert citations[0].parsed.title == "Real Paper"
    assert citations[0].citation_id == "ref_001"


@pytest.mark.asyncio
async def test_missing_title_key_skipped():
    payload = [
        {"authors": ["A"], "year": 2020, "context": "c", "claim": "k"},
        {"title": "Kept", "authors": [], "year": 2020, "context": "c", "claim": "k"},
    ]
    extractor = CitationExtractor(_FakeLLM(payload))
    citations = await extractor.extract("dummy")
    assert len(citations) == 1
    assert citations[0].parsed.title == "Kept"


@pytest.mark.asyncio
async def test_empty_payload_returns_empty_list():
    extractor = CitationExtractor(_FakeLLM([]))
    assert await extractor.extract("dummy") == []


@pytest.mark.asyncio
async def test_llm_failure_returns_empty_list():
    class BoomLLM:
        async def chat_json(self, prompt, system=None):
            raise RuntimeError("boom")

    extractor = CitationExtractor(BoomLLM())
    assert await extractor.extract("dummy") == []
