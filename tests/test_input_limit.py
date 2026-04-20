"""测试输入长度上限检查"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agents.orchestrator import HalluRefPipeline
from src.models.schemas import InputTooLargeError


@pytest.fixture
def pipeline():
    cfg = {
        "llm": {"base_url": "http://x", "api_key": "x", "model": "x"},
        "detection": {"max_input_chars": 100},
    }
    p = HalluRefPipeline(config=cfg)
    yield p


@pytest.mark.asyncio
async def test_run_rejects_oversize(pipeline):
    text = "a" * 200
    with pytest.raises(InputTooLargeError) as exc_info:
        await pipeline.run(text)
    assert "200" in str(exc_info.value)
    assert "100" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_streaming_rejects_oversize(pipeline):
    text = "a" * 200
    with pytest.raises(InputTooLargeError):
        async for _ in pipeline.run_streaming(text):
            pass


def test_api_returns_413():
    from app.api import app
    with TestClient(app) as client:
        # 默认配置上限 100000，构造超限文本
        text = "a" * 100001
        resp = client.post("/api/detect", json={"text": text})
    assert resp.status_code == 413
    assert "上限" in resp.json()["message"]
