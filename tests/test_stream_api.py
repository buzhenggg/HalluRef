"""POST /api/detect/stream SSE 流式接口测试"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app
from src.models.schemas import (
    Citation,
    CitationVerdict,
    ContentCheckResult,
    ContentConsistency,
    DetectionReport,
    HallucinationType,
    MatchConfidence,
    MetadataComparisonResult,
    ParsedCitation,
    RetrievalResult,
)


# ── 测试辅助 ──────────────────────────────────────────────


def _make_citation(cid: str, title: str) -> Citation:
    return Citation(
        citation_id=cid,
        raw_text=f"[{cid}] {title}",
        parsed=ParsedCitation(title=title, authors=["Author"]),
    )


def _make_verdict(cid: str, verdict: HallucinationType) -> CitationVerdict:
    return CitationVerdict(
        citation_id=cid,
        raw_text=f"[{cid}]",
        verdict=verdict,
        confidence=0.9,
        evidence="test evidence",
    )


def _parse_sse_events(text: str) -> list[dict]:
    """解析 SSE 文本为事件列表"""
    events = []
    current_event = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}
    if current_event:
        events.append(current_event)
    return events


# ── 测试类 ────────────────────────────────────────────────


class TestStreamEndpoint:
    """SSE 流式检测接口"""

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self):
        """空文本应返回 400"""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/detect/stream", json={"text": ""})
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_stream_events_order(self):
        """应按 extraction_done → citation_verdict(s) → report_complete 顺序推送"""
        citations = [_make_citation("ref_001", "Paper A")]
        retrieval = RetrievalResult(
            citation_id="ref_001", found=False, confidence=MatchConfidence.NONE
        )
        metadata = MetadataComparisonResult(citation_id="ref_001")
        content = ContentCheckResult(citation_id="ref_001")
        verdict = _make_verdict("ref_001", HallucinationType.FABRICATED)

        with patch.object(app.state, "pipeline", create=True) as mock_pipeline:
            mock_pipeline.agent1_extractor = AsyncMock()
            mock_pipeline.agent1_extractor.extract = AsyncMock(return_value=citations)
            mock_pipeline.agent2_retriever = AsyncMock()
            mock_pipeline.agent2_retriever.verify = AsyncMock(return_value=retrieval)
            mock_pipeline.agent3_comparator = AsyncMock()
            mock_pipeline.agent3_comparator.compare = lambda c, r: metadata
            mock_pipeline.agent4_checker = AsyncMock()
            mock_pipeline.agent4_checker.check = AsyncMock(return_value=content)
            mock_pipeline.agent5_reporter = AsyncMock()
            mock_pipeline.agent5_reporter._classify = lambda c, r, m, cc: verdict

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/detect/stream",
                    json={"text": "some academic text"},
                )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = _parse_sse_events(resp.text)
            event_types = [e["event"] for e in events]

            assert event_types[0] == "extraction_done"
            assert "citation_verdict" in event_types
            assert event_types[-1] == "report_complete"

    @pytest.mark.asyncio
    async def test_extraction_done_contains_total(self):
        """extraction_done 事件应包含引用总数"""
        citations = [
            _make_citation("ref_001", "Paper A"),
            _make_citation("ref_002", "Paper B"),
        ]
        retrieval = RetrievalResult(
            citation_id="ref_001", found=False, confidence=MatchConfidence.NONE
        )
        metadata = MetadataComparisonResult(citation_id="ref_001")
        content = ContentCheckResult(citation_id="ref_001")
        verdict = _make_verdict("ref_001", HallucinationType.FABRICATED)

        with patch.object(app.state, "pipeline", create=True) as mock_pipeline:
            mock_pipeline.agent1_extractor = AsyncMock()
            mock_pipeline.agent1_extractor.extract = AsyncMock(return_value=citations)
            mock_pipeline.agent2_retriever = AsyncMock()
            mock_pipeline.agent2_retriever.verify = AsyncMock(return_value=retrieval)
            mock_pipeline.agent3_comparator = AsyncMock()
            mock_pipeline.agent3_comparator.compare = lambda c, r: metadata
            mock_pipeline.agent4_checker = AsyncMock()
            mock_pipeline.agent4_checker.check = AsyncMock(return_value=content)
            mock_pipeline.agent5_reporter = AsyncMock()
            mock_pipeline.agent5_reporter._classify = lambda c, r, m, cc: verdict

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/detect/stream",
                    json={"text": "some academic text"},
                )

            events = _parse_sse_events(resp.text)
            extraction = next(e for e in events if e["event"] == "extraction_done")
            assert extraction["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_each_citation_yields_verdict_event(self):
        """每条引用都应产生一个 citation_verdict 事件"""
        citations = [
            _make_citation("ref_001", "Paper A"),
            _make_citation("ref_002", "Paper B"),
            _make_citation("ref_003", "Paper C"),
        ]
        retrieval = RetrievalResult(
            citation_id="x", found=False, confidence=MatchConfidence.NONE
        )
        metadata = MetadataComparisonResult(citation_id="x")
        content = ContentCheckResult(citation_id="x")

        def make_verdict_for(c, r, m, cc):
            return _make_verdict(c.citation_id, HallucinationType.FABRICATED)

        with patch.object(app.state, "pipeline", create=True) as mock_pipeline:
            mock_pipeline.agent1_extractor = AsyncMock()
            mock_pipeline.agent1_extractor.extract = AsyncMock(return_value=citations)
            mock_pipeline.agent2_retriever = AsyncMock()
            mock_pipeline.agent2_retriever.verify = AsyncMock(return_value=retrieval)
            mock_pipeline.agent3_comparator = AsyncMock()
            mock_pipeline.agent3_comparator.compare = lambda c, r: metadata
            mock_pipeline.agent4_checker = AsyncMock()
            mock_pipeline.agent4_checker.check = AsyncMock(return_value=content)
            mock_pipeline.agent5_reporter = AsyncMock()
            mock_pipeline.agent5_reporter._classify = make_verdict_for

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/detect/stream",
                    json={"text": "some academic text"},
                )

            events = _parse_sse_events(resp.text)
            verdict_events = [e for e in events if e["event"] == "citation_verdict"]
            assert len(verdict_events) == 3
            ids = {e["data"]["citation_id"] for e in verdict_events}
            assert ids == {"ref_001", "ref_002", "ref_003"}

    @pytest.mark.asyncio
    async def test_report_complete_has_statistics(self):
        """report_complete 事件应包含汇总统计"""
        citations = [_make_citation("ref_001", "Paper A")]
        retrieval = RetrievalResult(
            citation_id="ref_001", found=False, confidence=MatchConfidence.NONE
        )
        metadata = MetadataComparisonResult(citation_id="ref_001")
        content = ContentCheckResult(citation_id="ref_001")
        verdict = _make_verdict("ref_001", HallucinationType.FABRICATED)

        with patch.object(app.state, "pipeline", create=True) as mock_pipeline:
            mock_pipeline.agent1_extractor = AsyncMock()
            mock_pipeline.agent1_extractor.extract = AsyncMock(return_value=citations)
            mock_pipeline.agent2_retriever = AsyncMock()
            mock_pipeline.agent2_retriever.verify = AsyncMock(return_value=retrieval)
            mock_pipeline.agent3_comparator = AsyncMock()
            mock_pipeline.agent3_comparator.compare = lambda c, r: metadata
            mock_pipeline.agent4_checker = AsyncMock()
            mock_pipeline.agent4_checker.check = AsyncMock(return_value=content)
            mock_pipeline.agent5_reporter = AsyncMock()
            mock_pipeline.agent5_reporter._classify = lambda c, r, m, cc: verdict

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/detect/stream",
                    json={"text": "some academic text"},
                )

            events = _parse_sse_events(resp.text)
            report_evt = next(e for e in events if e["event"] == "report_complete")
            data = report_evt["data"]
            assert data["total_citations"] == 1
            assert data["fabricated"] == 1
