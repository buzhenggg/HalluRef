"""Agent 4: 内容一致性核查智能体

验证 LLM 对论文观点的引述是否忠实于原文。
策略: 获取论文摘要 → LLM 执行 NLI 判断。
"""

from __future__ import annotations

import json

from loguru import logger

from src.models.schemas import (
    Citation,
    ContentCheckResult,
    ContentConsistency,
    RetrievalResult,
)
from src.utils.llm_client import LLMClient

_SYSTEM_PROMPT = "你是一个学术事实核查专家。你的任务是判断引用声明是否忠实于论文原文。"

_CHECK_PROMPT = """\
给定以下论文摘要和引用声明, 判断引用声明是否忠实于原文:

论文标题: {title}
论文摘要: {abstract}

引用声明: {claim}

请从以下选项中选择:
1. CONSISTENT - 引用声明与论文内容一致
2. INCONSISTENT - 引用声明与论文内容矛盾
3. UNVERIFIABLE - 仅凭摘要无法判断
4. EXAGGERATED - 引用声明夸大了论文结论

请以JSON格式返回:
```json
{{
  "consistency": "CONSISTENT | INCONSISTENT | UNVERIFIABLE | EXAGGERATED",
  "reasoning": "简要说明判断理由 (1-2句话)"
}}
```
"""


class ContentChecker:
    """内容一致性核查 Agent"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def check(
        self, citation: Citation, retrieval: RetrievalResult
    ) -> ContentCheckResult | None:
        """核查单条引用的内容一致性"""
        cid = citation.citation_id

        # 前置条件: 论文必须找到且有摘要, 引用必须有 claim
        if not retrieval.found or not retrieval.best_match:
            return ContentCheckResult(
                citation_id=cid,
                consistency=ContentConsistency.UNVERIFIABLE,
                reasoning="论文未检索到, 无法进行内容核查",
            )

        abstract = retrieval.best_match.abstract or ""
        claim = (citation.claim or "").strip()

        # 无具体声明 → 跳过内容核查，返回 None 让决策树忽略此环节
        if not claim:
            logger.info(f"[Agent4] {cid}: 无具体声明, 跳过内容核查")
            return None

        if not abstract:
            # 摘要缺失不再判 UNVERIFIABLE: 元数据已能证明论文真实存在,
            # 内容核查只是无法执行, 由 Agent 5 按元数据情况落到 VERIFIED / VERIFIED_MINOR
            logger.info(f"[Agent4] {cid}: 未获取到摘要, 跳过内容核查")
            return None

        # LLM NLI 判断
        prompt = _CHECK_PROMPT.format(
            title=retrieval.best_match.title,
            abstract=abstract,
            claim=claim,
        )

        try:
            result = await self.llm.chat_json(prompt, system=_SYSTEM_PROMPT)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[Agent4] {cid}: LLM NLI failed: {e}")
            return ContentCheckResult(
                citation_id=cid,
                consistency=ContentConsistency.UNVERIFIABLE,
                reasoning=f"LLM 核查失败: {e}",
                claim=claim,
                abstract=abstract,
            )

        consistency_str = result.get("consistency", "UNVERIFIABLE").upper()
        try:
            consistency = ContentConsistency(consistency_str)
        except ValueError:
            consistency = ContentConsistency.UNVERIFIABLE

        reasoning = result.get("reasoning", "")

        logger.info(f"[Agent4] {cid}: {consistency.value} — {reasoning[:80]}")

        return ContentCheckResult(
            citation_id=cid,
            consistency=consistency,
            reasoning=reasoning,
            claim=claim,
            abstract=abstract,
        )

    async def check_batch(
        self,
        citations: list[Citation],
        retrievals: list[RetrievalResult],
    ) -> list[ContentCheckResult | None]:
        """批量核查"""
        import asyncio

        retrieval_map = {r.citation_id: r for r in retrievals}
        tasks = []
        for c in citations:
            r = retrieval_map.get(c.citation_id)
            if r:
                tasks.append(self.check(c, r))
            else:
                tasks.append(self._empty_result(c.citation_id))
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _empty_result(cid: str) -> ContentCheckResult:
        return ContentCheckResult(
            citation_id=cid,
            consistency=ContentConsistency.UNVERIFIABLE,
            reasoning="无检索结果",
        )
