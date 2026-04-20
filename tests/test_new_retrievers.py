"""单元测试: SemanticScholar / arXiv / Serper 检索器

通过 monkeypatch _request_with_retry, 用预置响应验证解析逻辑,
不依赖真实网络。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.retrievers.arxiv import ArxivRetriever
from src.retrievers.semantic_scholar import SemanticScholarRetriever
from src.retrievers.serper import SerperRetriever


# ── 工具: 构造假 Response ─────────────────────────────────


def _fake_json_resp(payload: dict):
    return SimpleNamespace(
        json=lambda: payload,
        text=json.dumps(payload),
        status_code=200,
    )


def _fake_text_resp(text: str):
    return SimpleNamespace(
        json=lambda: {},
        text=text,
        status_code=200,
    )


# ───────────────────────── Semantic Scholar ─────────────────────────


class TestSemanticScholar:
    @pytest.mark.asyncio
    async def test_parse_title_search(self, monkeypatch):
        payload = {
            "data": [
                {
                    "title": "Attention Is All You Need",
                    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "externalIds": {"DOI": "10.5555/3295222.3295349"},
                    "abstract": "We propose a new simple network architecture...",
                    "tldr": {"text": "Transformer architecture."},
                    "url": "https://www.semanticscholar.org/paper/xxx",
                }
            ]
        }
        r = SemanticScholarRetriever()

        async def fake_req(method, url, **kw):
            assert "search" in url
            assert kw["params"]["query"] == "Attention Is All You Need"
            return _fake_json_resp(payload)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)

        papers = await r.search_by_title("Attention Is All You Need")
        assert len(papers) == 1
        p = papers[0]
        assert p.source == "semantic_scholar"
        assert p.title == "Attention Is All You Need"
        assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert p.year == 2017
        assert p.venue == "NeurIPS"
        assert p.doi == "10.5555/3295222.3295349"
        assert "network architecture" in p.abstract

    @pytest.mark.asyncio
    async def test_tldr_fallback(self, monkeypatch):
        payload = {
            "data": [
                {
                    "title": "X",
                    "authors": [],
                    "year": 2020,
                    "tldr": {"text": "TLDR text"},
                }
            ]
        }
        r = SemanticScholarRetriever()
        monkeypatch.setattr(r, "_request_with_retry", lambda *a, **kw: _fake_async(payload))

        papers = await r.search_by_title("X")
        assert papers[0].abstract == "TLDR text"

    @pytest.mark.asyncio
    async def test_search_by_author_year(self, monkeypatch):
        captured = {}

        async def fake_req(method, url, **kw):
            captured.update(kw["params"])
            return _fake_json_resp({"data": []})

        r = SemanticScholarRetriever()
        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        await r.search_by_author_year(["Vaswani", "Shazeer"], 2017)
        assert captured["query"] == "Vaswani Shazeer"
        assert captured["year"] == "2017"

    @pytest.mark.asyncio
    async def test_api_key_in_header(self, monkeypatch):
        r = SemanticScholarRetriever(api_key="abc")
        captured = {}

        async def fake_req(method, url, **kw):
            captured.update(kw["headers"])
            return _fake_json_resp({"data": []})

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        await r.search_by_title("X")
        assert captured.get("x-api-key") == "abc"

    def test_api_key_required_for_configuration(self):
        assert SemanticScholarRetriever(api_key="abc").is_configured() is True
        assert SemanticScholarRetriever(api_key="").is_configured() is False
        assert SemanticScholarRetriever(api_key=None).is_configured() is False

    @pytest.mark.asyncio
    async def test_empty_data(self, monkeypatch):
        r = SemanticScholarRetriever()

        async def fake_req(method, url, **kw):
            return _fake_json_resp({"data": []})

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        papers = await r.search_by_title("nonexistent")
        assert papers == []


# 辅助: 让被替换的 _request_with_retry 仍是协程
async def _fake_async(payload):
    return _fake_json_resp(payload)


# ───────────────────────── arXiv ─────────────────────────


_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2106.09685v1</id>
    <published>2021-06-17T00:00:00Z</published>
    <title>LoRA: Low-Rank Adaptation of Large Language Models</title>
    <summary>We propose Low-Rank Adaptation, or LoRA...</summary>
    <author><name>Edward J. Hu</name></author>
    <author><name>Yelong Shen</name></author>
    <arxiv:doi>10.48550/arXiv.2106.09685</arxiv:doi>
    <arxiv:journal_ref>ICLR 2022</arxiv:journal_ref>
  </entry>
</feed>
"""


class TestArxiv:
    @pytest.mark.asyncio
    async def test_parse_title_search(self, monkeypatch):
        r = ArxivRetriever()

        async def fake_req(method, url, **kw):
            assert "search_query" in kw["params"]
            assert "ti:" in kw["params"]["search_query"]
            return _fake_text_resp(_ARXIV_XML)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)

        papers = await r.search_by_title("LoRA")
        assert len(papers) == 1
        p = papers[0]
        assert p.source == "arxiv"
        assert p.title.startswith("LoRA")
        assert p.authors == ["Edward J. Hu", "Yelong Shen"]
        assert p.year == 2021
        assert p.doi == "10.48550/arXiv.2106.09685"
        assert p.venue == "ICLR 2022"
        assert "Low-Rank Adaptation" in p.abstract
        assert p.url == "http://arxiv.org/abs/2106.09685v1"

    @pytest.mark.asyncio
    async def test_search_by_author_year_filters_year(self, monkeypatch):
        r = ArxivRetriever()

        async def fake_req(method, url, **kw):
            return _fake_text_resp(_ARXIV_XML)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)

        # 命中: 2021
        ok = await r.search_by_author_year(["Edward Hu"], 2021)
        assert len(ok) == 1
        # 不命中: 2099
        miss = await r.search_by_author_year(["Edward Hu"], 2099)
        assert miss == []

    @pytest.mark.asyncio
    async def test_search_by_author_year_no_authors(self, monkeypatch):
        r = ArxivRetriever()
        result = await r.search_by_author_year([], 2021)
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_xml(self, monkeypatch):
        r = ArxivRetriever()

        async def fake_req(method, url, **kw):
            return _fake_text_resp("<<not xml>>")

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        papers = await r.search_by_title("anything")
        assert papers == []

    @pytest.mark.asyncio
    async def test_global_cooldown_serializes_arxiv_requests(self, monkeypatch):
        ArxivRetriever._global_last_request_time = 0.0
        ArxivRetriever._global_lock = None
        ArxivRetriever._global_request_interval = 0.05

        r1 = ArxivRetriever()
        r2 = ArxivRetriever()
        starts = []

        async def gated_request(retriever):
            await retriever._rate_limit()
            starts.append(asyncio.get_event_loop().time())

        try:
            await asyncio.gather(gated_request(r1), gated_request(r2))
        finally:
            ArxivRetriever._global_request_interval = 3.5
            ArxivRetriever._global_last_request_time = 0.0
            ArxivRetriever._global_lock = None

        assert len(starts) == 2
        assert starts[1] - starts[0] >= 0.045


# ───────────────────────── Serper ─────────────────────────


class TestSerper:
    @pytest.mark.asyncio
    async def test_parse_search(self, monkeypatch):
        # Serper search API response shape: generic organic results.
        payload = {
            "organic": [
                {
                    "title": "[1706.03762] Attention Is All You Need - arXiv",
                    "link": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture, the Transformer...",
                    "date": "Jun 12, 2017",
                    "position": 1,
                }
            ]
        }
        r = SerperRetriever(api_key="dummy", enrich_links=False)

        async def fake_req(method, url, **kw):
            assert method == "POST"
            assert url == "https://google.serper.dev/search"
            assert kw["headers"]["X-API-KEY"] == "dummy"
            body = json.loads(kw["content"])
            assert body["q"] == "Attention Is All You Need"
            return _fake_json_resp(payload)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)

        papers = await r.search_by_title("Attention Is All You Need")
        assert len(papers) == 1
        p = papers[0]
        assert p.source == "serper"
        assert "Attention Is All You Need" in p.title
        assert p.authors == []  # Web Search 不含结构化作者
        assert p.year == 2017   # 从 date "Jun 12, 2017" 提取
        assert p.url == "https://arxiv.org/abs/1706.03762"
        assert "network architecture" in p.abstract

    @pytest.mark.asyncio
    async def test_year_parsed_from_snippet_when_no_date(self, monkeypatch):
        payload = {
            "organic": [
                {
                    "title": "X",
                    "snippet": "Published in Some Journal in 2019, this paper...",
                }
            ]
        }
        r = SerperRetriever(api_key="k", enrich_links=False)
        monkeypatch.setattr(r, "_request_with_retry", lambda *a, **kw: _fake_async(payload))
        papers = await r.search_by_title("X")
        assert papers[0].year == 2019

    @pytest.mark.asyncio
    async def test_search_by_author_year_query(self, monkeypatch):
        captured = {}

        async def fake_req(method, url, **kw):
            captured["body"] = json.loads(kw["content"])
            return _fake_json_resp({"organic": []})

        r = SerperRetriever(api_key="k", enrich_links=False)
        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        await r.search_by_author_year(["Vaswani", "Shazeer"], 2017)
        assert captured["body"]["q"] == "Vaswani Shazeer 2017"

    @pytest.mark.asyncio
    async def test_empty_organic(self, monkeypatch):
        r = SerperRetriever(api_key="k", enrich_links=False)
        monkeypatch.setattr(r, "_request_with_retry", lambda *a, **kw: _fake_async({"organic": []}))
        papers = await r.search_by_title("nothing")
        assert papers == []

    def test_is_configured(self):
        assert SerperRetriever(api_key="x").is_configured() is True
        assert SerperRetriever(api_key=None).is_configured() is False
        assert SerperRetriever(api_key="").is_configured() is False

    @pytest.mark.asyncio
    async def test_scholar_mode_uses_scholar_endpoint(self, monkeypatch):
        payload = {"organic": []}
        r = SerperRetriever(api_key="dummy", enrich_links=False, search_type="scholar")

        async def fake_req(method, url, **kw):
            assert url == "https://google.serper.dev/scholar"
            return _fake_json_resp(payload)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        assert await r.search_by_title("Attention Is All You Need") == []
        assert r.source_name == "serper_scholar"


class TestSerpApi:
    @pytest.mark.asyncio
    async def test_parse_search(self, monkeypatch):
        from src.retrievers.serpapi import SerpApiRetriever

        payload = {
            "organic_results": [
                {
                    "title": "Attention Is All You Need - arXiv",
                    "link": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture...",
                    "date": "2017",
                }
            ]
        }
        r = SerpApiRetriever(api_key="dummy", enrich_links=False)

        async def fake_req(method, url, **kw):
            assert method == "GET"
            assert url == "https://serpapi.com/search.json"
            assert kw["params"]["engine"] == "google"
            assert kw["params"]["api_key"] == "dummy"
            assert kw["params"]["q"] == "test query"
            return _fake_json_resp(payload)

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        papers = await r.search_by_title("test query")
        assert len(papers) == 1
        p = papers[0]
        assert p.source == "serpapi"
        assert p.year == 2017
        assert p.url.startswith("https://arxiv.org")

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        from src.retrievers.serpapi import SerpApiRetriever
        r = SerpApiRetriever(api_key="k", enrich_links=False)
        monkeypatch.setattr(r, "_request_with_retry", lambda *a, **kw: _fake_async({}))
        assert await r.search_by_title("x") == []

    def test_is_configured(self):
        from src.retrievers.serpapi import SerpApiRetriever
        assert SerpApiRetriever(api_key="x").is_configured() is True
        assert SerpApiRetriever(api_key=None).is_configured() is False

    @pytest.mark.asyncio
    async def test_scholar_mode_uses_google_scholar_engine(self, monkeypatch):
        from src.retrievers.serpapi import SerpApiRetriever

        r = SerpApiRetriever(api_key="dummy", enrich_links=False, search_type="scholar")

        async def fake_req(method, url, **kw):
            assert kw["params"]["engine"] == "google_scholar"
            return _fake_json_resp({"organic_results": []})

        monkeypatch.setattr(r, "_request_with_retry", fake_req)
        assert await r.search_by_title("Attention Is All You Need") == []
        assert r.source_name == "serpapi_scholar"
