"""LLM fallback tests for author metadata comparison."""

import pytest

from src.agents.metadata_comparator import MetadataComparator
from src.models.schemas import (
    Citation,
    FieldMatchStatus,
    MatchConfidence,
    ParsedCitation,
    RetrievalResult,
    RetrievedPaper,
)


class FakeLLM:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def chat_json(self, prompt: str, system: str = "", temperature=None):
        self.calls.append((prompt, system))
        if self.error:
            raise self.error
        return self.payload


def _citation(authors: list[str]) -> Citation:
    return Citation(
        citation_id="ref_001",
        raw_text="Example citation",
        parsed=ParsedCitation(
            title="Example Paper",
            authors=authors,
            year=2024,
            venue="Example Venue",
        ),
    )


def _retrieval(authors: list[str]) -> RetrievalResult:
    return RetrievalResult(
        citation_id="ref_001",
        found=True,
        confidence=MatchConfidence.HIGH,
        best_match=RetrievedPaper(
            source="test",
            title="Example Paper",
            authors=authors,
            year=2024,
            venue="Example Venue",
        ),
    )


def _author_field(metadata):
    return next(field for field in metadata.fields if field.field == "authors")


@pytest.mark.asyncio
async def test_llm_fallback_marks_ambiguous_authors_as_match():
    llm = FakeLLM({
        "is_consistent": True,
        "reasoning": "The abbreviated names correspond to the full names.",
    })
    comparator = MetadataComparator(llm=llm)

    metadata = await comparator.compare_async(
        _citation(["A. Example", "B. Writer"]),
        _retrieval(["Alice Example", "Brenda Writer", "Carlos Extra"]),
    )
    author_field = _author_field(metadata)

    assert author_field.status == FieldMatchStatus.MATCH
    assert author_field.similarity == 0.95
    assert len(llm.calls) == 1
    assert "List A:" in llm.calls[0][0]
    assert "List B:" in llm.calls[0][0]


@pytest.mark.asyncio
async def test_llm_fallback_marks_author_count_mismatch_as_mismatch():
    llm = FakeLLM({
        "is_consistent": False,
        "reasoning": "The author counts differ and neither list uses et al.",
    })
    comparator = MetadataComparator(llm=llm)

    metadata = await comparator.compare_async(
        _citation(["Alice Example", "Brenda Writer"]),
        _retrieval(["Alice Example", "Carlos Extra", "Dana Other"]),
    )
    author_field = _author_field(metadata)

    assert author_field.status == FieldMatchStatus.MISMATCH
    assert author_field.similarity == 0.0


@pytest.mark.asyncio
async def test_llm_fallback_invalid_json_uses_rule_score():
    llm = FakeLLM({"unexpected": True})
    comparator = MetadataComparator(llm=llm)

    metadata = await comparator.compare_async(
        _citation(["Alice Example", "Brenda Writer"]),
        _retrieval(["Alice Example", "Carlos Extra", "Dana Other"]),
    )
    author_field = _author_field(metadata)

    assert author_field.status == FieldMatchStatus.MISMATCH
    assert author_field.similarity == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_clear_rule_match_does_not_call_llm():
    llm = FakeLLM({
        "is_consistent": False,
        "reasoning": "This should not be used.",
    })
    comparator = MetadataComparator(llm=llm)

    metadata = await comparator.compare_async(
        _citation(["Z. Guo", "M. Schlichtkrull", "A. Vlachos"]),
        _retrieval(["Zhijiang Guo", "Michael Schlichtkrull", "Andreas Vlachos"]),
    )
    author_field = _author_field(metadata)

    assert author_field.status == FieldMatchStatus.MATCH
    assert len(llm.calls) == 0
