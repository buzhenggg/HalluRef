"""无 claim 引用应跳过内容一致性检查"""

import pytest

from src.agents.content_checker import ContentChecker
from src.agents.report_generator import ReportGenerator
from src.models.schemas import (
    Citation,
    CitationVerdict,
    ContentCheckResult,
    ContentConsistency,
    HallucinationType,
    MatchConfidence,
    MetadataComparisonResult,
    ParsedCitation,
    RetrievalResult,
    RetrievedPaper,
)


def _make_citation(cid: str, claim: str = "") -> Citation:
    return Citation(
        citation_id=cid,
        raw_text=f"[{cid}]",
        parsed=ParsedCitation(title="Some Paper", authors=["Author"]),
        claim=claim,
    )


def _make_retrieval(cid: str, found: bool = True) -> RetrievalResult:
    if not found:
        return RetrievalResult(citation_id=cid, found=False, confidence=MatchConfidence.NONE)
    return RetrievalResult(
        citation_id=cid,
        found=True,
        confidence=MatchConfidence.HIGH,
        best_match=RetrievedPaper(
            source="openalex",
            title="Some Paper",
            authors=["Author"],
            year=2023,
            abstract="This paper proposes a method.",
        ),
    )


# ── ContentChecker 测试 ────────────────────────────────


class TestContentCheckerSkip:
    """无 claim 时 ContentChecker 应返回 None 而非调用 LLM"""

    @pytest.mark.asyncio
    async def test_no_claim_returns_none(self):
        """claim 为空时应返回 None"""
        checker = ContentChecker(llm=None)  # llm 不应被调用
        citation = _make_citation("ref_001", claim="")
        retrieval = _make_retrieval("ref_001", found=True)

        result = await checker.check(citation, retrieval)
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_claim_returns_none(self):
        """claim 仅含空白时应返回 None"""
        checker = ContentChecker(llm=None)
        citation = _make_citation("ref_001", claim="   ")
        retrieval = _make_retrieval("ref_001", found=True)

        result = await checker.check(citation, retrieval)
        assert result is None

    @pytest.mark.asyncio
    async def test_not_found_still_returns_unverifiable(self):
        """论文未找到时仍应返回 UNVERIFIABLE（无论有无 claim）"""
        checker = ContentChecker(llm=None)
        citation = _make_citation("ref_001", claim="some claim")
        retrieval = _make_retrieval("ref_001", found=False)

        result = await checker.check(citation, retrieval)
        assert result is not None
        assert result.consistency == ContentConsistency.UNVERIFIABLE


# ── ReportGenerator 测试 ───────────────────────────────


class TestReportGeneratorNoClaim:
    """无 claim 引用在决策树中应跳过内容检查，直接判定"""

    def test_no_claim_found_metadata_ok_becomes_verified(self):
        """论文存在、元数据一致、无 claim → VERIFIED"""
        reporter = ReportGenerator()
        citation = _make_citation("ref_001", claim="")
        retrieval = _make_retrieval("ref_001", found=True)
        metadata = MetadataComparisonResult(citation_id="ref_001", has_major_mismatch=False)

        verdict = reporter._classify(citation, retrieval, metadata, content=None)

        assert verdict.verdict == HallucinationType.VERIFIED

    def test_no_claim_metadata_error_still_detected(self):
        """无 claim 但元数据有严重不匹配 → METADATA_ERROR"""
        reporter = ReportGenerator()
        citation = _make_citation("ref_001", claim="")
        retrieval = _make_retrieval("ref_001", found=True)
        metadata = MetadataComparisonResult(
            citation_id="ref_001", has_major_mismatch=True, mismatch_count=2
        )

        verdict = reporter._classify(citation, retrieval, metadata, content=None)

        assert verdict.verdict == HallucinationType.METADATA_ERROR

    def test_no_claim_not_found_still_fabricated(self):
        """无 claim 但论文未找到 → FABRICATED"""
        reporter = ReportGenerator()
        citation = _make_citation("ref_001", claim="")
        retrieval = _make_retrieval("ref_001", found=False)

        verdict = reporter._classify(citation, retrieval, metadata=None, content=None)

        assert verdict.verdict == HallucinationType.FABRICATED
