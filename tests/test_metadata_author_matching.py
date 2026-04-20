"""Metadata comparison tests for citation-style author initials."""

from src.agents.metadata_comparator import MetadataComparator
from src.agents.report_generator import ReportGenerator
from src.models.schemas import (
    Citation,
    FieldMatchStatus,
    HallucinationType,
    MatchConfidence,
    ParsedCitation,
    RetrievalResult,
    RetrievedPaper,
)


def test_initial_author_citation_does_not_create_metadata_error():
    citation = Citation(
        citation_id="ref_001",
        raw_text=(
            "Z. Guo, M. Schlichtkrull, A. Vlachos, "
            "A survey on automated factchecking, Transactions of the Association "
            "for Computational Linguistics 10 (2022) 178-206."
        ),
        parsed=ParsedCitation(
            title="A Survey on Automated Fact-Checking",
            authors=["Z. Guo", "M. Schlichtkrull", "A. Vlachos"],
            year=2022,
            venue="Transactions of the Association for Computational Linguistics",
        ),
    )
    retrieval = RetrievalResult(
        citation_id="ref_001",
        found=True,
        confidence=MatchConfidence.HIGH,
        best_match=RetrievedPaper(
            source="crossref",
            title="A Survey on Automated Fact-Checking",
            authors=["Zhijiang Guo", "Michael Schlichtkrull", "Andreas Vlachos"],
            year=2022,
            venue="Transactions of the Association for Computational Linguistics",
        ),
    )

    metadata = MetadataComparator().compare(citation, retrieval)
    author_field = next(f for f in metadata.fields if f.field == "authors")
    verdict = ReportGenerator().classify_one(citation, retrieval, metadata)

    assert author_field.status == FieldMatchStatus.MATCH
    assert verdict.verdict == HallucinationType.VERIFIED


def test_et_al_author_citation_does_not_create_metadata_error():
    citation = Citation(
        citation_id="ref_001",
        raw_text="Vaswani et al. Attention Is All You Need. 2017.",
        parsed=ParsedCitation(
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            year=2017,
            venue="NeurIPS",
        ),
    )
    retrieval = RetrievalResult(
        citation_id="ref_001",
        found=True,
        confidence=MatchConfidence.HIGH,
        best_match=RetrievedPaper(
            source="crossref",
            title="Attention Is All You Need",
            authors=[
                "Ashish Vaswani",
                "Noam Shazeer",
                "Niki Parmar",
                "Jakob Uszkoreit",
                "Llion Jones",
                "Aidan N. Gomez",
                "Lukasz Kaiser",
                "Illia Polosukhin",
            ],
            year=2017,
            venue="NeurIPS",
        ),
    )

    metadata = MetadataComparator().compare(citation, retrieval)
    author_field = next(f for f in metadata.fields if f.field == "authors")
    verdict = ReportGenerator().classify_one(citation, retrieval, metadata)

    assert author_field.status == FieldMatchStatus.MATCH
    assert verdict.verdict == HallucinationType.VERIFIED
