"""FastAPI 后端 — 提供 REST API + 静态前端"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.orchestrator import HalluRefPipeline, build_retrieval_config_status
from src.models.schemas import CitationVerdict, DetectionReport, InputTooLargeError
from src.utils.config import load_config
from src.utils import document_text

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


class ExtractFileResponse(BaseModel):
    filename: str
    content_type: str
    text: str
    char_count: int
    max_input_chars: int
    over_limit: bool


def _detection_limits(app_inst: FastAPI) -> tuple[int, int]:
    cfg = load_config()
    det_cfg = cfg.get("detection", {})
    max_input_chars = getattr(
        app_inst.state,
        "max_input_chars",
        det_cfg.get("max_input_chars", 100000),
    )
    max_upload_bytes = getattr(
        app_inst.state,
        "max_upload_bytes",
        det_cfg.get("max_upload_bytes", 10 * 1024 * 1024),
    )
    return int(max_input_chars), int(max_upload_bytes)


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """检测文本中的幻觉引用"""
    try:
        report = await pipeline.run(req.text)
    except InputTooLargeError as e:
        return JSONResponse(status_code=413, content={"message": str(e)})
    return DetectResponse(report=report)


@app.post("/api/extract-file", response_model=ExtractFileResponse)
async def extract_file(request: Request, file: UploadFile = File(...)):
    """Extract plain text from an uploaded PDF or DOCX without running detection."""
    filename = file.filename or ""
    content_type = file.content_type or ""
    max_input_chars, max_upload_bytes = _detection_limits(request.app)

    content = await file.read(max_upload_bytes + 1)
    if not content:
        return JSONResponse(status_code=400, content={"message": "Uploaded file is empty."})
    if len(content) > max_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "message": (
                    f"Uploaded file is {len(content)} bytes, exceeding the "
                    f"{max_upload_bytes} byte limit."
                )
            },
        )

    try:
        text = document_text.extract_document_text(filename, content_type, content)
    except document_text.UnsupportedDocumentError as exc:
        return JSONResponse(status_code=415, content={"message": str(exc)})
    except document_text.EmptyDocumentTextError as exc:
        return JSONResponse(status_code=422, content={"message": str(exc)})
    except document_text.DocumentParseError as exc:
        return JSONResponse(status_code=422, content={"message": str(exc)})
    except document_text.DocumentTextError as exc:
        return JSONResponse(status_code=422, content={"message": str(exc)})

    if not text.strip():
        return JSONResponse(
            status_code=422,
            content={"message": "No extractable text found in uploaded document."},
        )

    char_count = len(text)
    return ExtractFileResponse(
        filename=filename,
        content_type=content_type,
        text=text,
        char_count=char_count,
        max_input_chars=max_input_chars,
        over_limit=char_count > max_input_chars,
    )


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


@app.get("/api/retrieval/config")
async def retrieval_config():
    cfg = load_config()
    status = build_retrieval_config_status(cfg.get("retriever", {}))
    max_input_chars, max_upload_bytes = _detection_limits(app)
    status["detection"] = {
        "max_input_chars": max_input_chars,
        "max_upload_bytes": max_upload_bytes,
    }
    return status


# 静态文件 & SPA fallback
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
