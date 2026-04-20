"""Retrieval configuration status API tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app


@pytest.mark.asyncio
async def test_retrieval_config_endpoint_returns_tiers():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/retrieval/config")

    assert resp.status_code == 200
    data = resp.json()
    assert "policy" in data
    assert data["detection"]["max_input_chars"] == 100000
    assert data["detection"]["max_upload_bytes"] >= 1
    tiers = {tier["tier"]: tier for tier in data["tiers"]}
    assert set(tiers) == {"academic", "scholar_search", "google_scholar_direct"}
    assert tiers["academic"]["providers"]
    assert tiers["scholar_search"]["providers"]
    assert tiers["google_scholar_direct"]["providers"][0]["source"] == "google_scholar"
