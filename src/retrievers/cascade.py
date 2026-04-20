"""Cascade retriever with academic -> scholar API -> direct Scholar tiers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from src.models.schemas import MatchConfidence, RetrievedPaper
from src.retrievers.base import BaseRetriever
from src.utils.text_similarity import title_similarity


# 来源优先级: 数字越小越靠前 (用于同分时 tiebreak 与最终候选排序)
_SOURCE_PRIORITY = {
    "openalex": 0,
    "crossref": 1,
    "arxiv": 2,
    "semantic_scholar": 3,
    "serper_scholar": 4,
    "serpapi_scholar": 5,
    "google_scholar": 6,
}


@dataclass
class CascadeSearchResult:
    found: bool = False
    confidence: MatchConfidence = MatchConfidence.NONE
    best_match: RetrievedPaper | None = None
    score: float = 0.0
    hit_tier: str | None = None  # 'academic_primary' / 'academic_secondary' / 'scholar_search' / 'google_scholar_direct'
    candidates: list[RetrievedPaper] = field(default_factory=list)
    tiers_run: list[str] = field(default_factory=list)
    debug_log: str = ""


class CascadeRetriever:
    """级联检索器

    层级顺序:
        1. academic_primary (OpenAlex + CrossRef)
        2. academic_secondary (arXiv + Semantic Scholar)
        3. scholar_search
        4. google_scholar_direct

    """

    def __init__(
        self,
        openalex: BaseRetriever | None = None,
        crossref: BaseRetriever | None = None,
        arxiv: BaseRetriever | None = None,
        semantic_scholar: BaseRetriever | None = None,
        scholar_search: BaseRetriever | None = None,
        scholar_search_fallback: BaseRetriever | None = None,
        google_scholar_direct=None,
        title_exact_threshold: float = 0.95,
        title_fuzzy_threshold: float = 0.85,
        per_request_timeout: float = 30.0,
    ):
        self.exact_thresh = title_exact_threshold
        self.fuzzy_thresh = title_fuzzy_threshold
        self.per_request_timeout = per_request_timeout

        # ── 构建有效 tier 列表 ───────────────────────
        # 每个 tier: (tier_name, retrievers, parallel)
        self.tiers: list[tuple[str, list, bool]] = []

        # Tier 1: 学术 API 并行
        academic_primary = [
            r for r in (openalex, crossref)
            if r is not None and self._is_configured(r)
        ]
        if academic_primary:
            self.tiers.append(("academic_primary", academic_primary, True))

        academic_secondary = [
            r for r in (arxiv, semantic_scholar)
            if r is not None and self._is_configured(r)
        ]
        if academic_secondary:
            self.tiers.append(("academic_secondary", academic_secondary, True))

        # Tier 2: Scholar Search API, Serper preferred over SerpAPI.
        chosen_scholar = None
        if scholar_search is not None and self._is_configured(scholar_search):
            chosen_scholar = scholar_search
        elif scholar_search_fallback is not None and self._is_configured(scholar_search_fallback):
            chosen_scholar = scholar_search_fallback
        if chosen_scholar is not None:
            self.tiers.append(("scholar_search", [chosen_scholar], False))

        # Tier 3: direct Google Scholar HTTP crawler fallback.
        # If a Scholar API is configured, direct crawling stays disabled for this run.
        if (
            chosen_scholar is None
            and google_scholar_direct is not None
            and self._is_configured(google_scholar_direct)
        ):
            self.tiers.append(("google_scholar_direct", [google_scholar_direct], False))

        logger.info(
            f"[Cascade] active tiers: "
            f"{[(name, [r.source_name for r in rs]) for name, rs, _ in self.tiers]}"
        )

    @staticmethod
    def _is_configured(retriever) -> bool:
        check = getattr(retriever, "is_configured", None)
        return bool(check()) if callable(check) else True

    async def _safe_search(
        self, retriever, tier_name, title, authors, year
    ) -> tuple[list[RetrievedPaper], str]:
        """单检索器调用, 返回 (论文列表, 调试摘要)"""
        try:
            papers = await asyncio.wait_for(
                retriever.search(title=title, authors=authors, year=year),
                timeout=self.per_request_timeout,
            )
            detail = getattr(retriever, "last_search_debug", "") or (
                f"{len(papers)} candidates"
            )
            return papers, f"{tier_name}/{retriever.source_name}: {detail}"
        except asyncio.TimeoutError:
            logger.warning(f"[Cascade] {retriever.source_name} timeout")
            return [], f"{tier_name}/{retriever.source_name}: error (timeout)"
        except Exception as e:
            logger.warning(f"[Cascade] {retriever.source_name} error: {e}")
            return [], f"{tier_name}/{retriever.source_name}: error ({e})"

    def _confidence_for(self, score: float) -> MatchConfidence:
        if score >= self.exact_thresh:
            return MatchConfidence.HIGH
        if score >= self.fuzzy_thresh:
            return MatchConfidence.MEDIUM
        if score >= 0.6:
            return MatchConfidence.LOW
        return MatchConfidence.NONE

    def _pick_best(
        self, title: str, candidates: list[RetrievedPaper]
    ) -> tuple[RetrievedPaper | None, float]:
        """从候选中选最佳匹配; 同分时按 _SOURCE_PRIORITY 决出"""
        best: RetrievedPaper | None = None
        best_score = 0.0
        best_priority = 999

        for paper in candidates:
            if not paper.title or not title:
                continue
            score = title_similarity(title, paper.title)
            prio = _SOURCE_PRIORITY.get(paper.source, 999)
            if score > best_score or (score == best_score and prio < best_priority):
                best = paper
                best_score = score
                best_priority = prio

        return best, best_score

    async def search(
        self,
        title: str = "",
        authors: list[str] | None = None,
        year: int | None = None,
    ) -> CascadeSearchResult:
        """级联检索一个引用

        Returns: CascadeSearchResult
        """
        authors = authors or []
        result = CascadeSearchResult()
        debug_lines: list[str] = ["检索调试:"]

        if not title and not authors:
            logger.warning("[Cascade] empty query (no title/authors)")
            result.debug_log = "检索调试:\n- 空查询: 未提供 title/authors"
            return result

        for tier_name, retrievers, parallel in self.tiers:
            result.tiers_run.append(tier_name)
            logger.debug(
                f"[Cascade] tier '{tier_name}' "
                f"({'parallel' if parallel else 'serial'}, "
                f"{len(retrievers)} retriever(s))"
            )

            # ── 执行该 tier ────────────────
            if parallel:
                tier_results = await asyncio.gather(
                    *[
                        self._safe_search(r, tier_name, title, authors, year)
                        for r in retrievers
                    ]
                )
            else:
                tier_results = [
                    await self._safe_search(
                        retrievers[0], tier_name, title, authors, year
                    )
                ]

            tier_papers: list[RetrievedPaper] = []
            for papers, debug_line in tier_results:
                tier_papers.extend(papers)
                debug_lines.append(f"- {debug_line}")

            result.candidates.extend(tier_papers)

            # ── 评估匹配, 命中阈值即停 ─────
            best, score = self._pick_best(title, result.candidates)
            confidence = self._confidence_for(score)

            logger.info(
                f"[Cascade] tier '{tier_name}' done: "
                f"+{len(tier_papers)} candidates, best_score={score:.2f}, "
                f"confidence={confidence.value}"
            )

            if confidence in (MatchConfidence.HIGH, MatchConfidence.MEDIUM):
                # 截断候选, 标记命中
                result.found = True
                result.confidence = confidence
                result.best_match = best
                result.score = score
                result.hit_tier = tier_name
                # 候选按来源优先级稳定排序, 限 10 条
                result.candidates = sorted(
                    result.candidates,
                    key=lambda p: _SOURCE_PRIORITY.get(p.source, 999),
                )[:10]
                debug_lines.append(f"- 最终命中层: {tier_name}")
                result.debug_log = "\n".join(debug_lines)
                return result

        # ── 全部 tier 跑完仍未命中 ──
        best, score = self._pick_best(title, result.candidates)
        result.confidence = self._confidence_for(score)
        result.best_match = best
        result.score = score
        result.found = False
        result.candidates = sorted(
            result.candidates,
            key=lambda p: _SOURCE_PRIORITY.get(p.source, 999),
        )[:10]
        debug_lines.append("- 最终未命中: 所有 tier 均未达到阈值")
        result.debug_log = "\n".join(debug_lines)
        return result

    async def close(self):
        for _, retrievers, _ in self.tiers:
            for r in retrievers:
                try:
                    await r.close()
                except Exception:
                    pass
