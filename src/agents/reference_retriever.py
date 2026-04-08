"""Agent 2: 文献检索核查智能体

级联（Cascade）策略, 按 tier 顺序调用检索器, 任一 tier 命中阈值即停止,
后续 tier 不再发起请求, 节省 API 调用与时间:
    Tier 1 academic    OpenAlex + CrossRef     并行
    Tier 2 web_search  Serper / SerpAPI        二选一
    Tier 3 scholar     Google Scholar 直爬     兜底

实际级联逻辑由 ``CascadeRetriever`` 实现, 本 Agent 仅做适配 (输出 RetrievalResult)。
若未注入 cascade, 则回退到旧的"全部并发 + 可选 fallback"行为以保持向后兼容。
"""

from __future__ import annotations

import asyncio
import random

from loguru import logger

from src.models.schemas import (
    Citation,
    MatchConfidence,
    RetrievalResult,
    RetrievedPaper,
)
from src.retrievers.base import BaseRetriever
from src.utils.text_similarity import title_similarity


class ReferenceRetriever:
    """文献检索核查 Agent（多源级联, 命中即停）

    串行限流: 同一时刻只允许一条引用进入 verify, 上一条完成后还需
    等待一段 ``[interval_min, interval_max]`` 的随机间隔才放下一条进入,
    用于降低对外部检索 API 的访问压力, 同时避免固定间隔被识别为爬虫。
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever] | None = None,
        title_exact_threshold: float = 0.95,
        title_fuzzy_threshold: float = 0.85,
        max_concurrent: int = 5,
        fallback_retriever=None,
        cascade=None,
        interval_min: float = 1.0,
        interval_max: float = 2.0,
    ):
        self.retrievers = retrievers or []
        self.exact_thresh = title_exact_threshold
        self.fuzzy_thresh = title_fuzzy_threshold
        self.max_concurrent = max_concurrent
        self.fallback_retriever = fallback_retriever
        self.cascade = cascade  # CascadeRetriever 实例; 优先使用
        # 串行入口锁: 同一时刻只放一条引用进入 verify
        self._entry_lock = asyncio.Lock()
        self._last_release_time: float = 0.0
        self.interval_min = interval_min
        self.interval_max = interval_max

    async def _search_one(
        self, retriever: BaseRetriever, title: str, authors: list[str], year: int | None,
    ) -> list[RetrievedPaper]:
        """调用单个检索器, 失败时返回空列表而非抛异常"""
        try:
            return await retriever.search(title=title, authors=authors, year=year)
        except Exception as e:
            logger.warning(f"[Agent2] {retriever.source_name} error: {e}")
            return []

    def _find_best_match(
        self, title: str, candidates: list[RetrievedPaper]
    ) -> tuple[RetrievedPaper | None, float, MatchConfidence, bool]:
        """从候选论文中选出最佳匹配，返回 (best_match, best_score, confidence, found)"""
        best_match: RetrievedPaper | None = None
        best_score: float = 0.0

        for paper in candidates:
            if not paper.title or not title:
                continue
            score = title_similarity(title, paper.title)
            if score > best_score:
                best_score = score
                best_match = paper

        if best_score >= self.exact_thresh:
            confidence = MatchConfidence.HIGH
        elif best_score >= self.fuzzy_thresh:
            confidence = MatchConfidence.MEDIUM
        elif best_score >= 0.6:
            confidence = MatchConfidence.LOW
        else:
            confidence = MatchConfidence.NONE

        found = confidence in (MatchConfidence.HIGH, MatchConfidence.MEDIUM)
        return best_match, best_score, confidence, found

    async def verify(self, citation: Citation) -> RetrievalResult:
        """验证单条引用的论文是否存在

        通过 ``_entry_lock`` 保证全局串行: 同一时刻只放一条引用进入,
        上一条完成后还需等待 ``min_interval`` 秒才放下一条进入。
        """
        async with self._entry_lock:
            # 等够距离上一次释放的随机间隔 (避免固定节奏被识别为爬虫)
            target_interval = random.uniform(self.interval_min, self.interval_max)
            now = asyncio.get_event_loop().time()
            wait = target_interval - (now - self._last_release_time)
            if wait > 0:
                logger.debug(
                    f"[Agent2] {citation.citation_id}: wait {wait:.2f}s "
                    f"(target={target_interval:.2f}s)"
                )
                await asyncio.sleep(wait)
            try:
                return await self._verify_locked(citation)
            finally:
                self._last_release_time = asyncio.get_event_loop().time()

    async def _verify_locked(self, citation: Citation) -> RetrievalResult:
        """实际检索逻辑 (在锁内执行)"""
        title = citation.parsed.title
        authors = citation.parsed.authors
        year = citation.parsed.year

        if not title and not authors:
            logger.warning(f"[Agent2] {citation.citation_id}: no title or authors to search")
            return RetrievalResult(
                citation_id=citation.citation_id,
                found=False,
                confidence=MatchConfidence.NONE,
            )

        # ── 优先使用 CascadeRetriever (tier 级早停) ──
        if self.cascade is not None:
            cascade_result = await self.cascade.search(
                title=title, authors=authors, year=year,
            )
            logger.info(
                f"[Agent2] {citation.citation_id}: cascade hit_tier="
                f"{cascade_result.hit_tier}, tiers_run={cascade_result.tiers_run}, "
                f"score={cascade_result.score:.2f}, "
                f"confidence={cascade_result.confidence.value}, "
                f"found={cascade_result.found}"
            )
            return RetrievalResult(
                citation_id=citation.citation_id,
                found=cascade_result.found,
                confidence=cascade_result.confidence,
                best_match=cascade_result.best_match if cascade_result.found else None,
                all_candidates=cascade_result.candidates[:10],
            )

        # ── 兼容路径: 并发请求所有检索器 ──
        tasks = [
            self._search_one(retriever, title, authors, year)
            for retriever in self.retrievers
        ]
        results_per_retriever = await asyncio.gather(*tasks)

        all_candidates: list[RetrievedPaper] = []
        for papers in results_per_retriever:
            all_candidates.extend(papers)

        best_match, best_score, confidence, found = self._find_best_match(
            title, all_candidates
        )

        # ── Fallback: Google Scholar 补查 ──
        if not found and self.fallback_retriever is not None:
            logger.info(
                f"[Agent2] {citation.citation_id}: 主力检索未找到, "
                f"启动 Google Scholar 补查"
            )
            try:
                fallback_papers = await self.fallback_retriever.search(
                    title=title, authors=authors, year=year,
                )
                if fallback_papers:
                    all_candidates.extend(fallback_papers)
                    best_match, best_score, confidence, found = (
                        self._find_best_match(title, all_candidates)
                    )
                    if found:
                        logger.info(
                            f"[Agent2] {citation.citation_id}: "
                            f"Google Scholar 补查成功, score={best_score:.2f}"
                        )
            except Exception as e:
                logger.warning(
                    f"[Agent2] {citation.citation_id}: "
                    f"Google Scholar fallback error: {e}"
                )

        logger.info(
            f"[Agent2] {citation.citation_id}: "
            f"best_score={best_score:.2f}, confidence={confidence.value}, found={found}"
        )

        return RetrievalResult(
            citation_id=citation.citation_id,
            found=found,
            confidence=confidence,
            best_match=best_match if found else None,
            all_candidates=all_candidates[:10],
        )

    async def verify_batch(self, citations: list[Citation]) -> list[RetrievalResult]:
        """批量验证多条引用 — 使用信号量控制并发数"""
        if not citations:
            return []

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _limited_verify(citation: Citation) -> RetrievalResult:
            async with sem:
                return await self.verify(citation)

        return await asyncio.gather(*[_limited_verify(c) for c in citations])

    async def close(self):
        """关闭所有检索器连接"""
        if self.cascade is not None:
            try:
                await self.cascade.close()
            except Exception:
                pass
        for r in self.retrievers:
            await r.close()
        if self.fallback_retriever is not None:
            await self.fallback_retriever.close()
