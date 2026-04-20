"""报告决策树在移除内容一致性检测后的行为测试"""

from src.agents.report_generator import ReportGenerator
from src.models.schemas import (
    Citation,
    FieldComparison,
    FieldMatchStatus,
    HallucinationType,
    MatchConfidence,
    MetadataComparisonResult,
    ParsedCitation,
    RetrievalResult,
    RetrievedPaper,
)


def _make_citation(cid: str) -> Citation:
    return Citation(
        citation_id=cid,
        raw_text=f"[{cid}]",
        parsed=ParsedCitation(title="Some Paper", authors=["Author"]),
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
        ),
    )


class TestReportGeneratorWithoutContentCheck:
    def test_found_and_metadata_ok_becomes_verified(self):
        reporter = ReportGenerator()
        citation = _make_citation("ref_001")
        retrieval = _make_retrieval("ref_001", found=True)
        metadata = MetadataComparisonResult(citation_id="ref_001", has_major_mismatch=False)

        verdict = reporter._classify(citation, retrieval, metadata)

        assert verdict.verdict == HallucinationType.VERIFIED

    def test_metadata_error_still_detected(self):
        reporter = ReportGenerator()
        citation = _make_citation("ref_001")
        retrieval = _make_retrieval("ref_001", found=True)
        metadata = MetadataComparisonResult(
            citation_id="ref_001",
            fields=[
                FieldComparison(
                    field="title",
                    claimed="Wrong Title",
                    actual="Some Paper",
                    status=FieldMatchStatus.MISMATCH,
                    similarity=0.1,
                )
            ],
            has_major_mismatch=True,
            mismatch_count=1,
        )

        verdict = reporter._classify(citation, retrieval, metadata)

        assert verdict.verdict == HallucinationType.METADATA_ERROR

    def test_not_found_still_fabricated(self):
        reporter = ReportGenerator()
        citation = _make_citation("ref_001")
        retrieval = _make_retrieval("ref_001", found=False)

        verdict = reporter._classify(citation, retrieval, metadata=None)

        assert verdict.verdict == HallucinationType.FABRICATED
