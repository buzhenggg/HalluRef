"""Agent 1: CitationExtractor 单元测试"""

from __future__ import annotations

import pytest

from src.agents.citation_extractor import (
    CHUNK_SIZE,
    CONTEXT_SIZE,
    CitationExtractor,
)


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_prompt = None

    async def chat_json(self, prompt, system=None):
        self.calls += 1
        self.last_prompt = prompt
        return self.payload


class _SequenceLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.prompts = []

    async def chat_json(self, prompt, system=None):
        self.calls += 1
        self.prompts.append(prompt)
        payload = self.payloads[self.calls - 1]
        if isinstance(payload, Exception):
            raise payload
        return payload


def _prompt_section(prompt: str, name: str) -> str:
    start = f"--- {name} Start ---\n"
    end = f"\n--- {name} End ---"
    return prompt.split(start, 1)[1].split(end, 1)[0]


@pytest.mark.asyncio
async def test_skip_citation_without_title():
    payload = [
        {
            "authors": ["Alice"],
            "title": "Real Paper",
            "year": 2023,
            "context": "as shown in [1]",
        },
        {
            "authors": ["Bob"],
            "title": "",
            "year": 2022,
            "context": "see [2]",
        },
        {
            "authors": ["Carol"],
            "title": "   ",
            "year": 2021,
            "context": "see [3]",
        },
    ]
    extractor = CitationExtractor(_FakeLLM(payload))
    citations = await extractor.extract("dummy text")
    assert len(citations) == 1
    assert citations[0].parsed.title == "Real Paper"
    assert citations[0].citation_id == "ref_001"
    assert not hasattr(citations[0], "claim")


@pytest.mark.asyncio
async def test_missing_title_key_skipped():
    payload = [
        {"authors": ["A"], "year": 2020, "context": "c"},
        {"title": "Kept", "authors": [], "year": 2020, "context": "c"},
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


@pytest.mark.asyncio
async def test_prompt_no_longer_requests_claim_field():
    llm = _FakeLLM(
        [{"authors": ["Alice"], "title": "Real Paper", "year": 2023, "context": "ctx"}]
    )
    extractor = CitationExtractor(llm)

    await extractor.extract("dummy text")

    assert llm.last_prompt is not None
    assert "claim" not in llm.last_prompt


@pytest.mark.asyncio
async def test_prompt_explicitly_allows_no_citations_without_fabrication():
    llm = _FakeLLM([])
    extractor = CitationExtractor(llm)

    await extractor.extract("plain text without references")

    assert llm.last_prompt is not None
    assert "may contain no academic citations" in llm.last_prompt
    assert "Do not invent or fabricate citations" in llm.last_prompt
    assert "return an empty array `[]`" in llm.last_prompt


@pytest.mark.asyncio
async def test_short_text_does_not_chunk():
    llm = _FakeLLM(
        [{"authors": ["Alice"], "title": "Short Paper", "year": 2023, "context": "ctx"}]
    )
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("a" * (CHUNK_SIZE - 1))

    assert llm.calls == 1
    assert len(citations) == 1
    assert citations[0].parsed.title == "Short Paper"


@pytest.mark.asyncio
async def test_long_text_chunks_with_fixed_context():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(25000))
    llm = _SequenceLLM([[], [], []])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract(text)

    assert citations == []
    assert llm.calls == 3
    current_chunks = [_prompt_section(prompt, "Current Segment") for prompt in llm.prompts]
    prev_contexts = [_prompt_section(prompt, "Previous Context") for prompt in llm.prompts]
    next_contexts = [_prompt_section(prompt, "Following Context") for prompt in llm.prompts]
    assert current_chunks[0] == text[:CHUNK_SIZE]
    assert current_chunks[1] == text[CHUNK_SIZE:CHUNK_SIZE * 2]
    assert current_chunks[2] == text[CHUNK_SIZE * 2:CHUNK_SIZE * 3]
    assert prev_contexts[0] == ""
    assert prev_contexts[1] == text[CHUNK_SIZE - CONTEXT_SIZE:CHUNK_SIZE]
    assert prev_contexts[2] == text[CHUNK_SIZE * 2 - CONTEXT_SIZE:CHUNK_SIZE * 2]
    assert next_contexts[0] == text[CHUNK_SIZE:CHUNK_SIZE + CONTEXT_SIZE]
    assert next_contexts[1] == text[CHUNK_SIZE * 2:CHUNK_SIZE * 2 + CONTEXT_SIZE]
    assert next_contexts[2] == ""


@pytest.mark.asyncio
async def test_duplicate_titles_merge_context_and_keep_first_metadata():
    llm = _SequenceLLM([
        [
            {
                "authors": ["Alice"],
                "title": "Shared Paper",
                "year": None,
                "venue": None,
                "doi": None,
                "context": "first context",
            }
        ],
        [
            {
                "authors": ["Bob"],
                "title": "Shared Paper",
                "year": 2024,
                "venue": "ICML",
                "doi": "10.1/test",
                "context": "second context",
            }
        ],
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("x" * 11000)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.citation_id == "ref_001"
    assert citation.parsed.title == "Shared Paper"
    assert citation.parsed.authors == ["Alice"]
    assert citation.parsed.year == 2024
    assert citation.parsed.venue == "ICML"
    assert citation.parsed.doi == "10.1/test"
    assert citation.context == "first context | second context"
    assert citation.raw_text == "first context | second context"


@pytest.mark.asyncio
async def test_duplicate_titles_are_normalized_by_case_and_space():
    llm = _SequenceLLM([
        [{"authors": ["A"], "title": "  Deep   Learning  ", "year": 2020, "context": "a"}],
        [{"authors": ["B"], "title": "deep learning", "year": 2021, "context": "b"}],
        [{"authors": ["C"], "title": "Other Paper", "year": 2022, "context": "c"}],
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("y" * 25000)

    assert [c.parsed.title for c in citations] == ["Deep   Learning", "Other Paper"]
    assert [c.citation_id for c in citations] == ["ref_001", "ref_002"]
    assert citations[0].context == "a | b"


@pytest.mark.asyncio
async def test_failed_chunk_does_not_drop_other_chunks():
    llm = _SequenceLLM([
        RuntimeError("boom"),
        [{"authors": ["B"], "title": "Recovered", "year": 2024, "context": "ok"}],
        RuntimeError("boom again"),
        RuntimeError("boom final"),
        RuntimeError("boom exhausted"),
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("z" * 11000)

    assert llm.calls == 5
    assert len(citations) == 1
    assert citations[0].parsed.title == "Recovered"
    assert citations[0].citation_id == "ref_001"


@pytest.mark.asyncio
async def test_invalid_json_shape_retries_until_valid():
    llm = _SequenceLLM([
        {"title": "Not an array"},
        [{"authors": ["A"], "title": "Recovered", "year": 2024, "context": "ctx"}],
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("dummy")

    assert llm.calls == 2
    assert len(citations) == 1
    assert citations[0].parsed.title == "Recovered"


@pytest.mark.asyncio
async def test_invalid_field_type_retries_until_valid():
    llm = _SequenceLLM([
        [{"authors": "Alice", "title": "Bad", "year": 2024, "context": "ctx"}],
        [{"authors": ["Alice"], "title": "Good", "year": 2024, "context": "ctx"}],
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("dummy")

    assert llm.calls == 2
    assert len(citations) == 1
    assert citations[0].parsed.title == "Good"


@pytest.mark.asyncio
async def test_invalid_payload_returns_empty_after_three_attempts():
    llm = _SequenceLLM([
        {"title": "Not an array"},
        [{"authors": ["A"], "title": 123, "year": 2024, "context": "ctx"}],
        [{"authors": ["A"], "title": "Bad", "year": "2024", "context": "ctx"}],
    ])
    extractor = CitationExtractor(llm)

    citations = await extractor.extract("dummy")

    assert llm.calls == 3
    assert citations == []
