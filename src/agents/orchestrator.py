"""多 Agent 编排主控

串联 5 个 Agent 的流水线:
  文本 → 引用提取 → 文献检索 → 元数据比对 → 内容核查 → 综合报告
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.agents.citation_extractor import CitationExtractor
from src.agents.content_checker import ContentChecker
from src.agents.metadata_comparator import MetadataComparator
from src.agents.reference_retriever import ReferenceRetriever
from src.agents.report_generator import ReportGenerator
from src.models.schemas import (
    Citation,
    CitationVerdict,
    DetectionReport,
    HallucinationType,
    InputTooLargeError,
)
from src.utils.config import load_config
from src.utils.llm_client import LLMClient
from src.retrievers.openalex import OpenAlexRetriever
from src.retrievers.crossref import CrossRefRetriever
from src.retrievers.google_scholar import GoogleScholarRetriever
from src.retrievers.serper import SerperRetriever
from src.retrievers.serpapi import SerpApiRetriever
from src.retrievers.cascade import CascadeRetriever


class HalluRefPipeline:
    """HalluRef 检测流水线"""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        llm_cfg = cfg.get("llm", {})
        ret_cfg = cfg.get("retriever", {})
        sim_cfg = cfg.get("similarity", {})
        det_cfg = cfg.get("detection", {})

        # 初始化 LLM 客户端
        self.llm = LLMClient(
            base_url=llm_cfg.get("base_url"),
            api_key=llm_cfg.get("api_key"),
            model=llm_cfg.get("model", "gpt-4o"),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )

        # ── 构建级联检索器 (Tier 级早停) ──
        oa_cfg = ret_cfg.get("openalex", {})
        cr_cfg = ret_cfg.get("crossref", {})
        sp_cfg = ret_cfg.get("serper", {})
        sa_cfg = ret_cfg.get("serpapi", {})
        gs_cfg = ret_cfg.get("google_scholar", {})

        openalex = OpenAlexRetriever(
            mailto=oa_cfg.get("mailto"),
            timeout=oa_cfg.get("timeout", 15),
        )
        crossref = CrossRefRetriever(
            mailto=cr_cfg.get("mailto"),
            timeout=cr_cfg.get("timeout", 15),
        )
        serper = SerperRetriever(
            api_key=sp_cfg.get("api_key"),
            timeout=sp_cfg.get("timeout", 15),
            max_results=sp_cfg.get("max_results", 5),
        ) if sp_cfg.get("api_key") else None
        serpapi = SerpApiRetriever(
            api_key=sa_cfg.get("api_key"),
            timeout=sa_cfg.get("timeout", 20),
            max_results=sa_cfg.get("max_results", 5),
        ) if sa_cfg.get("api_key") else None
        scholar = None
        if gs_cfg.get("enabled", False):
            scholar = GoogleScholarRetriever(
                proxy=gs_cfg.get("proxy"),
                timeout=gs_cfg.get("timeout", 15),
                max_results=gs_cfg.get("max_results", 3),
            )

        cascade = CascadeRetriever(
            openalex=openalex,
            crossref=crossref,
            serper=serper,
            serpapi=serpapi,
            scholar=scholar,
            title_exact_threshold=sim_cfg.get("title_exact_threshold", 0.95),
            title_fuzzy_threshold=sim_cfg.get("title_fuzzy_threshold", 0.85),
        )

        # 初始化 5 个 Agent
        self.agent1_extractor = CitationExtractor(self.llm)
        self.agent2_retriever = ReferenceRetriever(
            cascade=cascade,
            title_exact_threshold=sim_cfg.get("title_exact_threshold", 0.95),
            title_fuzzy_threshold=sim_cfg.get("title_fuzzy_threshold", 0.85),
        )
        self.agent3_comparator = MetadataComparator(
            mismatch_threshold=det_cfg.get("metadata_mismatch_threshold", 2),
        )
        self.agent4_checker = ContentChecker(self.llm)
        self.agent5_reporter = ReportGenerator()

        # 输入长度上限
        self.max_input_chars = det_cfg.get("max_input_chars", 20000)

    def _check_input_length(self, text: str) -> None:
        if len(text) > self.max_input_chars:
            raise InputTooLargeError(len(text), self.max_input_chars)

    async def run(self, text: str) -> DetectionReport:
        """执行完整检测流水线

        Args:
            text: LLM 生成的包含引用的学术文本

        Returns:
            检测报告
        """
        logger.info("=" * 60)
        logger.info("HalluRef Pipeline START")
        logger.info("=" * 60)

        self._check_input_length(text)

        # Step 1: 引用提取
        logger.info("[Pipeline] Step 1/5: 引用提取")
        citations = await self.agent1_extractor.extract(text)
        if not citations:
            logger.warning("[Pipeline] 未提取到任何引用, 终止流水线")
            return DetectionReport()

        # Step 2-5: 调用流式接口收集结果
        verdicts: list[CitationVerdict] = []
        async for v in self._process_citations(citations):
            verdicts.append(v)

        # 聚合报告
        report = self.agent5_reporter.aggregate(verdicts)

        logger.info("=" * 60)
        logger.info("HalluRef Pipeline DONE")
        logger.info("=" * 60)

        return report

    async def _process_one(self, citation: Citation) -> CitationVerdict:
        """单条引用全链路处理 (Agent 2→3→4→5), 异常隔离, 永不抛出"""
        cid = citation.citation_id
        try:
            # 2. 检索
            retrieval = await self.agent2_retriever.verify(citation)
            # 3. 元数据比对 (同步)
            metadata = self.agent3_comparator.compare(citation, retrieval)
            # 4. 内容核查 (仅 found 时调用 LLM)
            content = None
            if retrieval.found:
                content = await self.agent4_checker.check(citation, retrieval)
            # 5. 单条研判
            return self.agent5_reporter.classify_one(
                citation, retrieval, metadata, content,
            )
        except Exception as e:
            logger.error(f"[Pipeline] {cid} error: {e}")
            return CitationVerdict(
                citation_id=cid,
                raw_text=citation.raw_text,
                verdict=HallucinationType.UNVERIFIABLE,
                confidence=0.0,
                evidence=f"处理失败: {e}",
                suggestion="该条引用处理过程中发生异常, 建议人工核实",
            )

    async def _process_citations(
        self, citations: list[Citation]
    ) -> AsyncIterator[CitationVerdict]:
        """对一批引用并发处理, 完成一条立即 yield 一条 (as_completed)

        资源限流由底层组件自管理:
            - 学术 API: BaseRetriever 的 request_interval
            - LLM 调用: LLMClient 的 Semaphore(max_concurrent)
        """
        if not citations:
            return
        tasks = [asyncio.create_task(self._process_one(c)) for c in citations]
        try:
            for fut in asyncio.as_completed(tasks):
                verdict = await fut
                yield verdict
        finally:
            # 任意 yield 中断时, 取消尚未完成的任务
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def run_streaming(
        self, text: str
    ) -> AsyncIterator[tuple[str, object]]:
        """流式版 pipeline, 每完成一阶段/一条引用立即 yield 事件

        Yields:
            (event_type, payload) 元组:
                ('extraction_done', list[Citation])
                ('citation_verdict', CitationVerdict)
                ('report_complete', DetectionReport)
        """
        logger.info("=" * 60)
        logger.info("HalluRef Pipeline START (streaming)")
        logger.info("=" * 60)

        self._check_input_length(text)

        # Step 1: 引用提取
        logger.info("[Pipeline] Step 1/5: 引用提取")
        citations = await self.agent1_extractor.extract(text)
        if not citations:
            logger.warning("[Pipeline] 未提取到任何引用")
            yield "report_complete", DetectionReport()
            return
        yield "extraction_done", citations

        # Step 2-5: 流式吐 verdict
        logger.info(
            f"[Pipeline] Step 2-5: per-citation streaming ({len(citations)} citations)"
        )
        verdicts: list[CitationVerdict] = []
        async for verdict in self._process_citations(citations):
            verdicts.append(verdict)
            yield "citation_verdict", verdict

        # 最终聚合
        report = self.agent5_reporter.aggregate(verdicts)
        logger.info("HalluRef Pipeline DONE (streaming)")
        yield "report_complete", report

    async def close(self):
        """清理资源"""
        await self.agent2_retriever.close()


async def _main():
    parser = argparse.ArgumentParser(description="HalluRef 幻觉引用检测")
    parser.add_argument("--input", "-i", required=True, help="输入文本文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出报告 JSON 路径")
    parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    args = parser.parse_args()

    # 读取输入
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    text = input_path.read_text(encoding="utf-8")

    # 加载配置
    config = None
    if args.config:
        config = load_config(args.config)

    # 运行流水线
    pipeline = HalluRefPipeline(config=config)
    try:
        report = await pipeline.run(text)
    except InputTooLargeError as e:
        logger.error(str(e))
        await pipeline.close()
        sys.exit(1)
    finally:
        await pipeline.close()

    # 输出报告
    report_json = report.model_dump_json(indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(report_json, encoding="utf-8")
        logger.info(f"报告已保存: {args.output}")
    else:
        print(report_json)


if __name__ == "__main__":
    asyncio.run(_main())
