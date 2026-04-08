"""Agent 5: 综合研判与报告生成智能体

汇总所有 Agent 结果, 对每条引用给出幻觉分类标签 + 置信度, 生成结构化检测报告。
"""

from __future__ import annotations

from loguru import logger

from src.models.schemas import (
    Citation,
    CitationVerdict,
    ContentCheckResult,
    ContentConsistency,
    DetectionReport,
    FieldMatchStatus,
    HallucinationType,
    MetadataComparisonResult,
    RetrievalResult,
)

# 关键字段: 这两个字段不匹配视为 METADATA_ERROR (信息篡改)
_KEY_FIELDS = {"title", "authors"}
# 次要字段: 仅这些字段不匹配, 视为 VERIFIED_MINOR (引用基本正确)
_MINOR_FIELDS = {"year", "venue"}


class ReportGenerator:
    """综合研判 Agent"""

    def classify_one(
        self,
        citation: Citation,
        retrieval: RetrievalResult | None,
        metadata: MetadataComparisonResult | None,
        content: ContentCheckResult | None,
    ) -> CitationVerdict:
        """对单条引用做研判 (供 per-citation pipeline 使用)"""
        return self._classify(citation, retrieval, metadata, content)

    def aggregate(self, verdicts: list[CitationVerdict]) -> DetectionReport:
        """从已有的逐条 verdict 聚合成最终报告"""
        report = DetectionReport(
            total_citations=len(verdicts),
            verified=sum(1 for v in verdicts if v.verdict == HallucinationType.VERIFIED),
            verified_minor=sum(1 for v in verdicts if v.verdict == HallucinationType.VERIFIED_MINOR),
            fabricated=sum(1 for v in verdicts if v.verdict == HallucinationType.FABRICATED),
            metadata_error=sum(1 for v in verdicts if v.verdict == HallucinationType.METADATA_ERROR),
            misrepresented=sum(1 for v in verdicts if v.verdict == HallucinationType.MISREPRESENTED),
            unverifiable=sum(1 for v in verdicts if v.verdict == HallucinationType.UNVERIFIABLE),
            details=verdicts,
        )
        logger.info(
            f"[Agent5] Report: total={report.total_citations}, "
            f"verified={report.verified}, verified_minor={report.verified_minor}, "
            f"fabricated={report.fabricated}, "
            f"metadata_error={report.metadata_error}, "
            f"misrepresented={report.misrepresented}, "
            f"unverifiable={report.unverifiable}"
        )
        return report

    def generate(
        self,
        citations: list[Citation],
        retrievals: list[RetrievalResult],
        metadata_results: list[MetadataComparisonResult],
        content_results: list[ContentCheckResult],
    ) -> DetectionReport:
        """汇总所有 Agent 结果, 生成检测报告"""

        # 建立索引
        retrieval_map = {r.citation_id: r for r in retrievals}
        metadata_map = {m.citation_id: m for m in metadata_results}
        content_map = {c.citation_id: c for c in content_results}

        verdicts: list[CitationVerdict] = []

        for citation in citations:
            cid = citation.citation_id
            retrieval = retrieval_map.get(cid)
            metadata = metadata_map.get(cid)
            content = content_map.get(cid)

            verdict = self._classify(citation, retrieval, metadata, content)
            verdicts.append(verdict)

        # 统计 (复用 aggregate)
        return self.aggregate(verdicts)

    def _classify(
        self,
        citation: Citation,
        retrieval: RetrievalResult | None,
        metadata: MetadataComparisonResult | None,
        content: ContentCheckResult | None,
    ) -> CitationVerdict:
        """对单条引用执行分类决策树

        判定优先级 (短路):
            1. 未检索到论文          → FABRICATED
            2. title 或 authors 错   → METADATA_ERROR
            3. 内容矛盾 / 夸大       → MISREPRESENTED
            4. year/venue 等小字段错 → VERIFIED_MINOR (若 LLM 内容存疑则 evidence 附注)
            5. 全部通过             → VERIFIED      (若 LLM 内容存疑则 evidence 附注)

        说明: LLM 内容核查返回 UNVERIFIABLE 时不再升级成 UNVERIFIABLE 标签
              (论文真实存在 + 关键元数据正确 = 引用本身没出错), 只在 evidence 中注明
              "内容核查存疑", 完整 LLM reasoning 仍随 content_check 字段返回前端供人工查看。
              UNVERIFIABLE 标签仅保留给 orchestrator 层的整条引用处理异常兜底。
        """

        # 1. 未找到 → FABRICATED
        if not retrieval or not retrieval.found:
            return CitationVerdict(
                citation_id=citation.citation_id,
                raw_text=citation.raw_text,
                verdict=HallucinationType.FABRICATED,
                confidence=0.9 if (retrieval and retrieval.all_candidates) else 0.95,
                evidence=self._fabricated_evidence(retrieval),
                suggestion="建议删除该引用或替换为真实文献",
                retrieval=retrieval,
                metadata=metadata,
                content_check=content,
            )

        # 2. 找到 → 关键字段 (title/authors) 检查
        key_mismatches, minor_mismatches = self._split_mismatches(metadata)
        if key_mismatches:
            return CitationVerdict(
                citation_id=citation.citation_id,
                raw_text=citation.raw_text,
                verdict=HallucinationType.METADATA_ERROR,
                confidence=0.85,
                evidence=f"关键字段不一致: {', '.join(key_mismatches)}",
                suggestion="请核实并修正引用的元数据信息",
                retrieval=retrieval,
                metadata=metadata,
                content_check=content,
            )

        # 3. 关键元数据 OK → 内容核查
        content_uncertain_note = ""
        if content:
            if content.consistency == ContentConsistency.INCONSISTENT:
                return CitationVerdict(
                    citation_id=citation.citation_id,
                    raw_text=citation.raw_text,
                    verdict=HallucinationType.MISREPRESENTED,
                    confidence=0.85,
                    evidence=f"内容不一致: {content.reasoning}",
                    suggestion="引用声明与论文实际内容不符, 请重新核实",
                    retrieval=retrieval,
                    metadata=metadata,
                    content_check=content,
                )
            if content.consistency == ContentConsistency.EXAGGERATED:
                return CitationVerdict(
                    citation_id=citation.citation_id,
                    raw_text=citation.raw_text,
                    verdict=HallucinationType.MISREPRESENTED,
                    confidence=0.75,
                    evidence=f"观点夸大: {content.reasoning}",
                    suggestion="引用声明夸大了论文结论, 建议调整表述",
                    retrieval=retrieval,
                    metadata=metadata,
                    content_check=content,
                )
            if content.consistency == ContentConsistency.UNVERIFIABLE:
                # 不升级为 UNVERIFIABLE 标签: 论文存在 + 关键元数据正确 = 引用本身正确
                # 仅在 evidence 中附注存疑, 并继续走 VERIFIED / VERIFIED_MINOR 分支
                content_uncertain_note = "; 内容声明存疑 (LLM 仅凭摘要无法定论, 建议人工核实)"

        # 4. 关键元数据 OK + 内容也没问题 → 看是否有 year/venue 小字段错
        if minor_mismatches:
            return CitationVerdict(
                citation_id=citation.citation_id,
                raw_text=citation.raw_text,
                verdict=HallucinationType.VERIFIED_MINOR,
                confidence=0.85,
                evidence=(
                    f"论文存在, 标题与作者匹配; 但 {', '.join(minor_mismatches)} "
                    f"字段与真实文献不符" + content_uncertain_note
                ),
                suggestion="建议修正这些字段的引用信息",
                retrieval=retrieval,
                metadata=metadata,
                content_check=content,
            )

        # 5. 一切正常 → VERIFIED
        if content is not None and content.consistency == ContentConsistency.CONSISTENT:
            evidence = "论文存在, 元数据一致, 内容核查通过"
        elif content is not None and content.consistency == ContentConsistency.UNVERIFIABLE:
            evidence = "论文存在, 元数据一致" + content_uncertain_note
        elif retrieval.best_match and not (retrieval.best_match.abstract or "").strip():
            evidence = "论文存在, 元数据一致 (摘要缺失, 跳过内容核查)"
        else:
            evidence = "论文存在, 元数据一致 (引用未包含具体声明, 跳过内容核查)"
        return CitationVerdict(
            citation_id=citation.citation_id,
            raw_text=citation.raw_text,
            verdict=HallucinationType.VERIFIED,
            confidence=0.9,
            evidence=evidence,
            suggestion="",
            retrieval=retrieval,
            metadata=metadata,
            content_check=content,
        )

    @staticmethod
    def _split_mismatches(
        metadata: MetadataComparisonResult | None,
    ) -> tuple[list[str], list[str]]:
        """把 MISMATCH 字段拆成关键字段 (title/authors) 与次要字段 (year/venue)"""
        if not metadata or not metadata.fields:
            return [], []
        key, minor = [], []
        for f in metadata.fields:
            if f.status != FieldMatchStatus.MISMATCH:
                continue
            if f.field in _KEY_FIELDS:
                key.append(f.field)
            elif f.field in _MINOR_FIELDS:
                minor.append(f.field)
        return key, minor

    @staticmethod
    def _fabricated_evidence(retrieval: RetrievalResult | None) -> str:
        if not retrieval:
            return "未执行检索"
        if not retrieval.all_candidates:
            return "在所有学术数据源中均未检索到该论文"
        return (
            f"在学术数据源中检索到 {len(retrieval.all_candidates)} 条候选, "
            f"但标题匹配度均低于阈值, 疑似捏造"
        )
