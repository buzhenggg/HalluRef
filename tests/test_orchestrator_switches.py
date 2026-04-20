"""编排层检索开关测试。"""

from __future__ import annotations

from src.agents import orchestrator


def _ret_cfg() -> dict:
    return {
        "openalex": {"mailto": "test@example.com"},
        "crossref": {"mailto": "test@example.com"},
        "arxiv": {"timeout": 15, "max_results": 5},
        "semantic_scholar": {"api_key": "semantic-key", "timeout": 15},
        "serper": {"api_key": "serper-key"},
        "serpapi": {"api_key": "serpapi-key"},
        "google_scholar": {"timeout": 15, "max_results": 3},
        "proxy": {"server": ""},
    }


def _sim_cfg() -> dict:
    return {
        "title_exact_threshold": 0.95,
        "title_fuzzy_threshold": 0.85,
    }


def _tier_sources(cascade) -> dict[str, list[str]]:
    return {
        tier_name: [r.source_name for r in retrievers]
        for tier_name, retrievers, _ in cascade.tiers
    }


def test_default_switches_build_three_step_retrieval():
    cascade = orchestrator.build_cascade_retriever(_ret_cfg(), _sim_cfg())

    tier_sources = _tier_sources(cascade)
    assert tier_sources["academic_primary"] == [
        "openalex",
        "crossref",
    ]
    assert tier_sources["academic_secondary"] == [
        "arxiv",
        "semantic_scholar",
    ]
    assert tier_sources["scholar_search"] == ["serper_scholar"]
    assert "google_scholar_direct" not in tier_sources


def test_semantic_scholar_requires_api_key(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    cfg = _ret_cfg()
    cfg["semantic_scholar"]["api_key"] = ""

    cascade = orchestrator.build_cascade_retriever(cfg, _sim_cfg())

    tier_sources = _tier_sources(cascade)
    assert tier_sources["academic_primary"] == ["openalex", "crossref"]
    assert tier_sources["academic_secondary"] == ["arxiv"]


def test_direct_scholar_enabled_only_when_no_scholar_api_key(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", True)
    cfg = _ret_cfg()
    cfg["serper"]["api_key"] = ""
    cfg["serpapi"]["api_key"] = ""

    cascade = orchestrator.build_cascade_retriever(cfg, _sim_cfg())

    tier_sources = _tier_sources(cascade)
    assert "scholar_search" not in tier_sources
    assert tier_sources["google_scholar_direct"] == ["google_scholar"]


def test_disable_academic_apis_removes_academic_tier(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", False)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", True)

    cascade = orchestrator.build_cascade_retriever(_ret_cfg(), _sim_cfg())

    tier_sources = _tier_sources(cascade)
    assert "academic" not in tier_sources
    assert "scholar_search" in tier_sources
    assert "google_scholar_direct" not in tier_sources


def test_disable_search_apis_keeps_direct_scholar_fallback(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", False)

    cascade = orchestrator.build_cascade_retriever(_ret_cfg(), _sim_cfg())

    tier_sources = _tier_sources(cascade)
    assert "scholar_search" not in tier_sources
    assert tier_sources["google_scholar_direct"] == ["google_scholar"]
    assert "academic_primary" in tier_sources
    assert "academic_secondary" in tier_sources


def test_direct_scholar_uses_proxy_and_config(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", True)
    cfg = _ret_cfg()
    cfg["serper"]["api_key"] = ""
    cfg["serpapi"]["api_key"] = ""
    cfg["proxy"]["server"] = "http://127.0.0.1:7890"
    cfg["google_scholar"] = {"timeout": 21, "max_results": 7}

    cascade = orchestrator.build_cascade_retriever(cfg, _sim_cfg())

    direct = dict((name, retrievers) for name, retrievers, _ in cascade.tiers)[
        "google_scholar_direct"
    ][0]
    assert direct.proxy == "http://127.0.0.1:7890"
    assert direct.timeout == 21
    assert direct.max_results == 7


def test_retrieval_config_status_marks_direct_standby_when_scholar_api_exists(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", True)

    status = orchestrator.build_retrieval_config_status(_ret_cfg())

    tiers = {tier["tier"]: tier for tier in status["tiers"]}
    academic = tiers["academic"]["providers"]
    assert any(
        p["source"] == "semantic_scholar" and p["status"] == "active"
        for p in academic
    )
    assert tiers["scholar_search"]["active"] is True
    assert tiers["google_scholar_direct"]["active"] is False
    direct = tiers["google_scholar_direct"]["providers"][0]
    assert direct["status"] == "standby"
    assert "Scholar API" in direct["message"]


def test_retrieval_config_status_marks_missing_search_keys_and_activates_direct(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    monkeypatch.setattr(orchestrator, "ENABLE_SEARCH_APIS", True)
    cfg = _ret_cfg()
    cfg["serper"]["api_key"] = ""
    cfg["serpapi"]["api_key"] = ""

    status = orchestrator.build_retrieval_config_status(cfg)

    tiers = {tier["tier"]: tier for tier in status["tiers"]}
    assert tiers["scholar_search"]["active"] is False
    assert {p["status"] for p in tiers["scholar_search"]["providers"]} == {"error"}
    assert tiers["google_scholar_direct"]["active"] is True
    assert tiers["google_scholar_direct"]["providers"][0]["status"] == "active"


def test_retrieval_config_status_marks_missing_semantic_scholar_key(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_ACADEMIC_APIS", True)
    cfg = _ret_cfg()
    cfg["semantic_scholar"]["api_key"] = ""

    status = orchestrator.build_retrieval_config_status(cfg)

    academic = {
        provider["source"]: provider
        for provider in status["tiers"][0]["providers"]
    }
    assert academic["semantic_scholar"]["configured"] is False
    assert academic["semantic_scholar"]["active"] is False
    assert academic["semantic_scholar"]["status"] == "error"
