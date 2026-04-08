"""Agent 5: ReportGenerator 决策树测试 (含 VERIFIED_MINOR 新分类)"""

from __future__ import annotations

from src.agents.report_generator import ReportGenerator
from src.models.schemas import (
    Citation,
    ContentCheckResult,
    ContentConsistency,
    FieldComparison,
    FieldMatchStatus,
    HallucinationType,
    MatchConfidence,
    MetadataComparisonResult,
    ParsedCitation,
    RetrievalResult,
    RetrievedPaper,
)


def _cit(cid="ref_001", title="Real Paper", authors=("A",), year=2023):
    return Citation(
        citation_id=cid,
        raw_text="ctx",
        parsed=ParsedCitation(
            title=title, authors=list(authors), year=year, venue="X"
        ),
        context="ctx",
        claim="some claim",
    )


def _retrieval(found=True, abstract="some abstract"):
    return RetrievalResult(
        citation_id="ref_001",
        found=found,
        confidence=MatchConfidence.HIGH if found else MatchConfidence.NONE,
        best_match=RetrievedPaper(
            title="Real Paper",
            authors=["A"],
            year=2023,
            venue="X",
            abstract=abstract,
            source="openalex",
        ) if found else None,
        all_candidates=[],
    )


def _meta(field_statuses):
    """field_statuses: list[(field, status)]"""
    fields = [
        FieldComparison(
            field=f, claimed="x", actual="y", status=s, similarity=0.0
        )
        for f, s in field_statuses
    ]
    mismatch = sum(1 for _, s in field_statuses if s == FieldMatchStatus.MISMATCH)
    return MetadataComparisonResult(
        citation_id="ref_001",
        fields=fields,
        mismatch_count=mismatch,
        has_major_mismatch=mismatch >= 2,
    )


def _content(consistency):
    return ContentCheckResult(
        citation_id="ref_001",
        consistency=consistency,
        reasoning="r",
        claim="c",
        abstract="a",
    )


rg = ReportGenerator()


def test_fabricated_when_not_found():
    v = rg.classify_one(_cit(), _retrieval(found=False), None, None)
    assert v.verdict == HallucinationType.FABRICATED


def test_metadata_error_when_title_mismatch():
    meta = _meta([("title", FieldMatchStatus.MISMATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.CONSISTENT))
    assert v.verdict == HallucinationType.METADATA_ERROR


def test_metadata_error_when_authors_mismatch():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.CONSISTENT))
    assert v.verdict == HallucinationType.METADATA_ERROR


def test_verified_minor_when_only_year_wrong():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.CONSISTENT))
    assert v.verdict == HallucinationType.VERIFIED_MINOR
    assert "year" in v.evidence


def test_verified_minor_when_year_and_venue_wrong():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH),
                  ("venue", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, None)
    assert v.verdict == HallucinationType.VERIFIED_MINOR


def test_verified_when_all_match():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.CONSISTENT))
    assert v.verdict == HallucinationType.VERIFIED


def test_verified_when_abstract_missing_and_no_content_check():
    """摘要缺失 → Agent4 返回 None → 仍判 VERIFIED, evidence 注明摘要缺失"""
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(abstract=""), meta, None)
    assert v.verdict == HallucinationType.VERIFIED
    assert "摘要缺失" in v.evidence
    assert v.verdict != HallucinationType.UNVERIFIABLE


def test_verified_when_no_claim_and_no_content_check():
    """引用无 claim → Agent4 返回 None → VERIFIED, evidence 注明跳过"""
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, None)
    assert v.verdict == HallucinationType.VERIFIED
    assert "跳过内容核查" in v.evidence
    assert v.verdict != HallucinationType.UNVERIFIABLE


def test_verified_minor_overrides_abstract_missing():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(abstract=""), meta, None)
    assert v.verdict == HallucinationType.VERIFIED_MINOR


def test_misrepresented_when_inconsistent():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.INCONSISTENT))
    assert v.verdict == HallucinationType.MISREPRESENTED


def test_content_unverifiable_falls_to_verified_with_note():
    """LLM 内容存疑 + 元数据全对 → VERIFIED, evidence 附注存疑"""
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.UNVERIFIABLE))
    assert v.verdict == HallucinationType.VERIFIED
    assert "存疑" in v.evidence
    # content_check 字段仍随结果返回, 前端可展示具体 reasoning
    assert v.content_check is not None
    assert v.content_check.consistency == ContentConsistency.UNVERIFIABLE


def test_content_unverifiable_with_minor_mismatch_falls_to_verified_minor():
    """LLM 内容存疑 + year 错 → VERIFIED_MINOR, evidence 同时附注 year 错和存疑"""
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta, _content(ContentConsistency.UNVERIFIABLE))
    assert v.verdict == HallucinationType.VERIFIED_MINOR
    assert "year" in v.evidence
    assert "存疑" in v.evidence


def test_aggregate_counts_verified_minor():
    verdicts = [
        rg.classify_one(_cit(), _retrieval(), _meta([("title", FieldMatchStatus.MATCH),
                                                     ("authors", FieldMatchStatus.MATCH),
                                                     ("year", FieldMatchStatus.MISMATCH)]),
                        _content(ContentConsistency.CONSISTENT)),
        rg.classify_one(_cit(), _retrieval(), _meta([("title", FieldMatchStatus.MATCH),
                                                     ("authors", FieldMatchStatus.MATCH)]),
                        _content(ContentConsistency.CONSISTENT)),
    ]
    report = rg.aggregate(verdicts)
    assert report.verified_minor == 1
    assert report.verified == 1
    assert report.total_citations == 2
