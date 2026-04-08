"""Agent 3: 元数据比对智能体

逐字段比对 LLM 生成的引用元数据与检索到的真实元数据。
"""

from __future__ import annotations

from loguru import logger

from src.models.schemas import (
    Citation,
    FieldComparison,
    FieldMatchStatus,
    MetadataComparisonResult,
    RetrievalResult,
)
from src.utils.name_matcher import author_list_similarity
from src.utils.text_similarity import title_similarity, venue_similarity


class MetadataComparator:
    """元数据比对 Agent"""

    def __init__(self, mismatch_threshold: int = 2):
        self.mismatch_threshold = mismatch_threshold

    def compare(
        self, citation: Citation, retrieval: RetrievalResult
    ) -> MetadataComparisonResult:
        """比对单条引用的元数据"""
        if not retrieval.found or not retrieval.best_match:
            return MetadataComparisonResult(
                citation_id=citation.citation_id,
                mismatch_count=0,
                has_major_mismatch=False,
            )

        claimed = citation.parsed
        actual = retrieval.best_match
        fields: list[FieldComparison] = []

        # ── 标题比对 ──
        if claimed.title:
            sim = title_similarity(claimed.title, actual.title)
            status = self._score_to_status(sim, exact=0.95, partial=0.80)
            fields.append(FieldComparison(
                field="title",
                claimed=claimed.title,
                actual=actual.title,
                status=status,
                similarity=sim,
            ))

        # ── 作者比对 ──
        if claimed.authors:
            score, _ = author_list_similarity(claimed.authors, actual.authors)
            # 放宽阈值: "et al." 缩写返回 0.8 → PARTIAL 而非 MISMATCH
            status = self._score_to_status(score, exact=0.80, partial=0.40)
            fields.append(FieldComparison(
                field="authors",
                claimed=", ".join(claimed.authors),
                actual=", ".join(actual.authors),
                status=status,
                similarity=score,
            ))

        # ── 年份比对 ──
        # 允许 ±1 年差异 (arXiv预印本 vs 正式发表常有差异)
        if claimed.year is not None and actual.year is not None:
            diff = abs(claimed.year - actual.year)
            if diff == 0:
                status = FieldMatchStatus.MATCH
                sim = 1.0
            elif diff == 1:
                status = FieldMatchStatus.PARTIAL
                sim = 0.8
            else:
                status = FieldMatchStatus.MISMATCH
                sim = 0.0
            fields.append(FieldComparison(
                field="year",
                claimed=str(claimed.year),
                actual=str(actual.year),
                status=status,
                similarity=sim,
            ))

        # ── venue 比对 ──
        if claimed.venue and actual.venue:
            sim = venue_similarity(claimed.venue, actual.venue)
            status = self._score_to_status(sim, exact=0.85, partial=0.60)
            fields.append(FieldComparison(
                field="venue",
                claimed=claimed.venue,
                actual=actual.venue,
                status=status,
                similarity=sim,
            ))

        mismatch_count = sum(
            1 for f in fields if f.status == FieldMatchStatus.MISMATCH
        )
        has_major = mismatch_count >= self.mismatch_threshold

        logger.info(
            f"[Agent3] {citation.citation_id}: "
            f"{len(fields)} fields compared, {mismatch_count} mismatches, "
            f"major={has_major}"
        )

        return MetadataComparisonResult(
            citation_id=citation.citation_id,
            fields=fields,
            mismatch_count=mismatch_count,
            has_major_mismatch=has_major,
        )

    def compare_batch(
        self,
        citations: list[Citation],
        retrievals: list[RetrievalResult],
    ) -> list[MetadataComparisonResult]:
        """批量比对"""
        retrieval_map = {r.citation_id: r for r in retrievals}
        results = []
        for c in citations:
            r = retrieval_map.get(c.citation_id)
            if r:
                results.append(self.compare(c, r))
            else:
                results.append(MetadataComparisonResult(citation_id=c.citation_id))
        return results

    @staticmethod
    def _score_to_status(
        score: float, exact: float = 0.95, partial: float = 0.80
    ) -> FieldMatchStatus:
        if score >= exact:
            return FieldMatchStatus.MATCH
        elif score >= partial:
            return FieldMatchStatus.PARTIAL
        else:
            return FieldMatchStatus.MISMATCH
