import json

import pytest

from agents import scout_arxiv, scout_github
from core.kb import KnowledgeBase
from fakes import FakeLLM, make_cfg, paper, repo


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


@pytest.fixture
def kb(cfg):
    return KnowledgeBase(cfg, "humanoid whole-body control")


def _patch_arxiv(monkeypatch, *, llm, search_results, neighbors=None, fetched=None):
    calls = {"search": [], "neighbors": [], "fetch": []}

    def fake_search(queries, *, per_query, category):
        calls["search"].append(queries)
        return [dict(p) for p in search_results]

    def fake_neighbors(arxiv_id, *, limit):
        calls["neighbors"].append(arxiv_id)
        return list((neighbors or {}).get(arxiv_id, []))

    def fake_fetch(ids, source="arxiv"):
        calls["fetch"].append((ids, source))
        return [paper(i, source=source) for i in ids if i in (fetched or ids)]

    monkeypatch.setattr(scout_arxiv, "complete", llm)
    monkeypatch.setattr(scout_arxiv.arxiv_source, "search", fake_search)
    monkeypatch.setattr(scout_arxiv.semantic_scholar, "neighbors", fake_neighbors)
    monkeypatch.setattr(scout_arxiv.arxiv_source, "fetch_by_ids", fake_fetch)
    return calls


def test_scout_arxiv_generates_queries_when_none(monkeypatch, kb, cfg):
    llm = FakeLLM({"arxiv_queries": 'Here: ["q1", "q2"]'})
    calls = _patch_arxiv(monkeypatch, llm=llm, search_results=[])

    assert scout_arxiv.run("humanoid whole-body control", kb, cfg, limit=10) == []
    assert calls["search"] == [["q1", "q2"]]
    call = llm.named("arxiv_queries")[0]
    assert "humanoid whole-body control" in call["prompt"]
    for word in ("Unitree G1", "AgiBot X2", "MuJoCo"):
        assert word in call["prompt"]
    assert call["model"] == cfg["models"]["fast"] and call["effort"] == "low"


def test_scout_arxiv_skips_known_and_caps(monkeypatch, kb, cfg):
    kb.add_paper(paper("A"), {"approaches": [], "capabilities": []})
    ranked = json.dumps(["D", "INVENTED", "B"])
    llm = FakeLLM({"arxiv_queries": '["q"]', "arxiv_rank": ranked})
    _patch_arxiv(monkeypatch, llm=llm, search_results=[paper(i) for i in "ABCDE"])

    got = scout_arxiv.run("t", kb, cfg, limit=2)
    assert [p["id"] for p in got] == ["D", "B"]

    got = scout_arxiv.run("t", kb, cfg, limit=3)
    assert [p["id"] for p in got] == ["D", "B", "C"]  # unranked candidates follow ranked ones

    calls_before = len(llm.named("arxiv_rank"))
    assert len(scout_arxiv.run("t", kb, cfg, limit=4)) == 4  # pool fits: no ranking call
    assert len(llm.named("arxiv_rank")) == calls_before
    rank_prompt = llm.named("arxiv_rank")[0]["prompt"]
    assert "Abstract of B." in rank_prompt and "Abstract of A." not in rank_prompt


def test_scout_arxiv_followup_uses_given_queries(monkeypatch, kb, cfg):
    llm = FakeLLM({})
    calls = _patch_arxiv(monkeypatch, llm=llm, search_results=[paper("X"), paper("Y")])

    got = scout_arxiv.run("t", kb, cfg, limit=5, queries=["x"])
    assert calls["search"] == [["x"]]
    assert llm.calls == []
    assert [p["source"] for p in got] == ["followup", "followup"]


def test_scout_arxiv_adds_semantic_scholar_neighbors(monkeypatch, kb, cfg):
    llm = FakeLLM({"arxiv_queries": '["q"]'})
    calls = _patch_arxiv(monkeypatch, llm=llm, search_results=[paper("A")], neighbors={"A": ["N1", "A"]})

    got = scout_arxiv.run("t", kb, cfg, limit=10)
    assert calls["neighbors"] == ["A"]
    assert calls["fetch"] == [(["N1"], "semantic_scholar")]
    assert {p["id"]: p["source"] for p in got} == {"A": "arxiv", "N1": "semantic_scholar"}


def test_scout_github_links_papers_and_filters(monkeypatch, kb, cfg):
    searches, pauses = [], []

    def fake_search(query, *, per_page, token):
        searches.append(query)
        if query == '"2401.00001" in:readme':
            return [repo("a/b")]
        return [repo("a/b"), repo("c/d"), repo("known/one")]

    kb.add_repo(repo("known/one"))
    llm = FakeLLM({"github_filter": json.dumps([{"full_name": "a/b", "why_relevant": "WBC code for G1."},
                                                 {"full_name": "zzz/invented", "why_relevant": "?"}])})
    monkeypatch.setattr(scout_github, "complete", llm)
    monkeypatch.setattr(scout_github.github_source, "search_repos", fake_search)
    monkeypatch.setattr(scout_github.github_source, "pause", lambda token: pauses.append(token))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    got = scout_github.run("t", kb, cfg, papers=[paper("2401.00001"), paper("local-notes")], queries=["wbc"])
    assert searches == ["wbc", '"2401.00001" in:readme']  # local papers have no arXiv id to look up
    assert len(pauses) == 1
    assert [r["full_name"] for r in got] == ["a/b"]
    assert got[0]["linked_papers"] == ["2401.00001"]
    assert got[0]["why_relevant"] == "WBC code for G1."
    filter_prompt = llm.named("github_filter")[0]["prompt"]
    assert "c/d" in filter_prompt and "known/one" not in filter_prompt


def test_scout_github_search_failure_is_skipped(monkeypatch, kb, cfg):
    def boom(query, *, per_page, token):
        raise OSError("network down")

    llm = FakeLLM({"github_queries": '["a", "b"]'})
    monkeypatch.setattr(scout_github, "complete", llm)
    monkeypatch.setattr(scout_github.github_source, "search_repos", boom)
    monkeypatch.setattr(scout_github.github_source, "pause", lambda token: None)
    assert scout_github.run("t", kb, cfg, papers=[]) == []
    assert llm.named("github_filter") == []
