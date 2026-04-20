"""Agent 3: 元数据比对模块

逐字段比对 LLM 生成的引用元数据与检索到的真实元数据。
"""

from __future__ import annotations

import json
from typing import Any

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

_AUTHOR_JUDGE_SYSTEM = (
    "You are a rigorous academic literature data expert, highly proficient in "
    "various academic citation formats (e.g., APA, IEEE, MLA, GB/T 7714) and "
    "cross-lingual name spelling conventions."
)

_AUTHOR_JUDGE_PROMPT = """\
# Role
You are a rigorous academic literature data expert, highly proficient in various academic citation formats (e.g., APA, IEEE, MLA, GB/T 7714) and cross-lingual name spelling conventions.

# Task
Please carefully compare the following two author lists (List A and List B) and determine whether they semantically refer to **the same group of authors**.

List A: {claimed_authors}
List B: {actual_authors}

# Judgment Rules (Very Important)
When determining consistency, you must strictly adhere to the following tolerance rules:
1. **Abbreviations vs. Full Names**: If one is a full first name (John) and the other is the corresponding initial (J. or J), they are considered consistent.
2. **First and Last Name Order**: Ignore the inversion of first and last names (e.g., "Smith, J. D." and "J. D. Smith" are considered consistent).
3. **Missing Middle Names**: If one list includes a middle initial (J. D. Smith) and the other only includes the first and last name (John Smith), they are considered consistent due to potential data source discrepancies.
4. **Special Characters and Pinyin**: Ignore differences in accent marks/diacritics (e.g., Müller and Muller are consistent). For Chinese authors, ignore the difference between full Pinyin and Pinyin initials (e.g., Zhang San and Zhang, S. are consistent).
5. **Separators and Conjunctions**: Completely ignore differences in punctuation and connecting words such as commas, semicolons, "and", "&", etc.
6. **"et al." Handling (Critical Rule)**:
   - If List A contains only the first few authors followed by "et al." (or equivalent), and List B provides the full list of authors, they are considered consistent as long as the first few authors match perfectly.
   - If the number of authors in the two lists is completely different, and neither contains "et al.", they are considered inconsistent.

# Output Format
Please output the results strictly in pure JSON format. Do not include any additional Markdown formatting (such as ```json) or explanatory text. The JSON must contain the following fields:
{{
  "is_consistent": true/false,
  "reasoning": "A brief and concise explanation of your reasoning, pointing out the key matching or mismatching points"
}}
"""

AUTHOR_RULE_MATCH_THRESHOLD = 0.80
AUTHOR_RULE_FALLBACK_THRESHOLD = 0.25
AUTHOR_LLM_MATCH_SCORE = 0.95


class MetadataComparator:
    """元数据比对 Agent"""

    def __init__(self, mismatch_threshold: int = 2, llm: Any | None = None):
        self.mismatch_threshold = mismatch_threshold
        self.llm = llm

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

    async def compare_async(
        self, citation: Citation, retrieval: RetrievalResult
    ) -> MetadataComparisonResult:
        """Compare metadata with optional LLM fallback for ambiguous authors."""
        if not retrieval.found or not retrieval.best_match:
            return MetadataComparisonResult(
                citation_id=citation.citation_id,
                mismatch_count=0,
                has_major_mismatch=False,
            )

        claimed = citation.parsed
        actual = retrieval.best_match
        fields: list[FieldComparison] = []

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

        if claimed.authors:
            score, _ = author_list_similarity(claimed.authors, actual.authors)
            score = await self._maybe_llm_author_score(
                claimed.authors,
                actual.authors,
                score,
            )
            status = self._score_to_status(
                score,
                exact=AUTHOR_RULE_MATCH_THRESHOLD,
                partial=0.40,
            )
            fields.append(FieldComparison(
                field="authors",
                claimed=", ".join(claimed.authors),
                actual=", ".join(actual.authors),
                status=status,
                similarity=score,
            ))

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

    async def _maybe_llm_author_score(
        self,
        claimed_authors: list[str],
        actual_authors: list[str],
        rule_score: float,
    ) -> float:
        if self.llm is None:
            return rule_score
        if not claimed_authors or not actual_authors:
            return rule_score
        if rule_score >= AUTHOR_RULE_MATCH_THRESHOLD:
            return rule_score
        if rule_score < AUTHOR_RULE_FALLBACK_THRESHOLD:
            return rule_score

        prompt = _AUTHOR_JUDGE_PROMPT.format(
            claimed_authors=json.dumps(claimed_authors, ensure_ascii=False),
            actual_authors=json.dumps(actual_authors, ensure_ascii=False),
        )
        try:
            payload = await self.llm.chat_json(
                prompt,
                system=_AUTHOR_JUDGE_SYSTEM,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"[Agent3] author LLM fallback failed: {exc}")
            return rule_score

        if not isinstance(payload, dict) or not isinstance(
            payload.get("is_consistent"), bool
        ):
            logger.warning(
                f"[Agent3] author LLM fallback returned invalid payload: {payload}"
            )
            return rule_score

        reasoning = payload.get("reasoning", "")
        logger.info(
            "[Agent3] author LLM fallback: "
            f"is_consistent={payload['is_consistent']}, reasoning={reasoning}"
        )
        return AUTHOR_LLM_MATCH_SCORE if payload["is_consistent"] else 0.0

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

    async def compare_batch_async(
        self,
        citations: list[Citation],
        retrievals: list[RetrievalResult],
    ) -> list[MetadataComparisonResult]:
        """Batch metadata comparison with optional LLM author fallback."""
        retrieval_map = {r.citation_id: r for r in retrievals}
        results = []
        for c in citations:
            r = retrieval_map.get(c.citation_id)
            if r:
                results.append(await self.compare_async(c, r))
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
