"""FastAPI 后端 — 提供 REST API + 静态前端"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.orchestrator import HalluRefPipeline
from src.models.schemas import CitationVerdict, DetectionReport, InputTooLargeError

pipeline: HalluRefPipeline | None = None

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    global pipeline
    pipeline = HalluRefPipeline()
    app_inst.state.pipeline = pipeline
    yield
    if pipeline:
        await pipeline.close()


app = FastAPI(
    title="HalluRef API",
    description="基于多智能体事实核查的 LLM 幻觉引用检测",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    report: DetectionReport


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """检测文本中的幻觉引用"""
    try:
        report = await pipeline.run(req.text)
    except InputTooLargeError as e:
        return JSONResponse(status_code=413, content={"message": str(e)})
    return DetectResponse(report=report)


def _sse_event(event: str, data: dict | str) -> dict:
    """构造 SSE 事件"""
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data,
    }


@app.post("/api/detect/stream")
async def detect_stream(request: Request):
    """SSE 流式检测 — 每完成一条引用立即推送结果"""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"message": "text 不能为空"})

    pipe = request.app.state.pipeline

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            async for event_type, payload in pipe.run_streaming(text):
                if event_type == "extraction_done":
                    citations = payload  # list[Citation]
                    yield _sse_event("extraction_done", {
                        "total": len(citations),
                        "citations": [
                            {"citation_id": c.citation_id, "raw_text": c.raw_text}
                            for c in citations
                        ],
                    })
                elif event_type == "citation_verdict":
                    verdict: CitationVerdict = payload
                    yield _sse_event(
                        "citation_verdict",
                        json.loads(verdict.model_dump_json(ensure_ascii=False)),
                    )
                elif event_type == "report_complete":
                    report: DetectionReport = payload
                    yield _sse_event(
                        "report_complete",
                        json.loads(report.model_dump_json(ensure_ascii=False)),
                    )

        except InputTooLargeError as e:
            yield _sse_event("error", {"message": str(e)})
        except Exception as exc:
            logger.error(f"[Stream] Pipeline error: {exc}")
            yield _sse_event("error", {"message": str(exc)})

    return EventSourceResponse(event_generator())


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# 静态文件 & SPA fallback
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/architecture")
async def architecture():
    return FileResponse(STATIC_DIR / "architecture.html")
