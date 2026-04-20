"""Agent 5: ReportGenerator 决策树测试 (含 VERIFIED_MINOR 新分类)"""

from __future__ import annotations

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


def _cit(cid="ref_001", title="Real Paper", authors=("A",), year=2023):
    return Citation(
        citation_id=cid,
        raw_text="ctx",
        parsed=ParsedCitation(
            title=title, authors=list(authors), year=year, venue="X"
        ),
        context="ctx",
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
        debug_log="",
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


rg = ReportGenerator()


def test_fabricated_when_not_found():
    v = rg.classify_one(_cit(), _retrieval(found=False), None)
    assert v.verdict == HallucinationType.FABRICATED


def test_fabricated_evidence_includes_debug_log():
    retrieval = _retrieval(found=False)
    retrieval.debug_log = (
        "检索调试:\n"
        "- academic/openalex: 0 candidates\n"
        "- academic/crossref: API error (timeout)"
    )
    v = rg.classify_one(_cit(), retrieval, None)
    assert v.verdict == HallucinationType.FABRICATED
    assert "检索调试" in v.evidence
    assert "openalex" in v.evidence
    assert "timeout" in v.evidence


def test_metadata_error_when_title_mismatch():
    meta = _meta([("title", FieldMatchStatus.MISMATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.METADATA_ERROR


def test_metadata_error_when_authors_mismatch():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.METADATA_ERROR


def test_unverifiable_when_retrieved_paper_missing_authors():
    retrieval = _retrieval()
    retrieval.best_match.authors = []

    v = rg.classify_one(_cit(), retrieval, _meta([("title", FieldMatchStatus.MATCH)]))

    assert v.verdict == HallucinationType.UNVERIFIABLE
    assert "authors" in v.evidence


def test_verified_minor_when_only_year_wrong():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.VERIFIED_MINOR
    assert "year" in v.evidence


def test_verified_minor_when_year_and_venue_wrong():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH),
                  ("venue", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.VERIFIED_MINOR


def test_verified_when_all_match():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.VERIFIED


def test_verified_when_metadata_match():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(abstract=""), meta)
    assert v.verdict == HallucinationType.VERIFIED
    assert "元数据一致" in v.evidence


def test_verified_when_claim_present_but_ignored():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH)])
    v = rg.classify_one(_cit(), _retrieval(), meta)
    assert v.verdict == HallucinationType.VERIFIED


def test_verified_minor_with_abstract_missing():
    meta = _meta([("title", FieldMatchStatus.MATCH),
                  ("authors", FieldMatchStatus.MATCH),
                  ("year", FieldMatchStatus.MISMATCH)])
    v = rg.classify_one(_cit(), _retrieval(abstract=""), meta)
    assert v.verdict == HallucinationType.VERIFIED_MINOR


def test_aggregate_counts_verified_minor():
    verdicts = [
        rg.classify_one(_cit(), _retrieval(), _meta([("title", FieldMatchStatus.MATCH),
                                                     ("authors", FieldMatchStatus.MATCH),
                                                     ("year", FieldMatchStatus.MISMATCH)]),
                        ),
        rg.classify_one(_cit(), _retrieval(), _meta([("title", FieldMatchStatus.MATCH),
                                                     ("authors", FieldMatchStatus.MATCH)])),
    ]
    report = rg.aggregate(verdicts)
    assert report.verified_minor == 1
    assert report.verified == 1
    assert report.total_citations == 2


def test_aggregate_sorts_details_by_citation_id():
    verdicts = [
        rg.classify_one(
            _cit(cid="ref_003"),
            _retrieval(),
            _meta([("title", FieldMatchStatus.MATCH), ("authors", FieldMatchStatus.MATCH)]),
        ),
        rg.classify_one(
            _cit(cid="ref_001"),
            _retrieval(),
            _meta([("title", FieldMatchStatus.MATCH), ("authors", FieldMatchStatus.MATCH)]),
        ),
        rg.classify_one(
            _cit(cid="ref_002"),
            _retrieval(),
            _meta([("title", FieldMatchStatus.MATCH), ("authors", FieldMatchStatus.MATCH)]),
        ),
    ]

    report = rg.aggregate(verdicts)

    assert [v.citation_id for v in report.details] == ["ref_001", "ref_002", "ref_003"]
