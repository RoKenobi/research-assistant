import json

import pytest

import pipeline
from agents.reader import ReaderError
from core.kb import KnowledgeBase
from core.llm import LLMUsageLimitError
from fakes import make_cfg, paper, repo

BD = {"problem": "p", "approaches": ["rl"], "capabilities": ["stairs"]}


class Recorder:
    def __init__(self):
        self.events = []


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Every agent replaced by a fake that records what it was asked."""
    rec = Recorder()
    cfg = make_cfg(tmp_path)
    state = {"initial": [paper("P1"), paper("P2")], "followup": [paper("F1")], "critic": [["more q"], []],
             "reader_fail": {}}

    def scout_arxiv(topic, kb, cfg_, *, limit, queries=None):
        rec.events.append(("scout_arxiv", limit, queries))
        offered = state["followup"] if queries else state["initial"]
        return [dict(p) for p in offered if not kb.has_paper(p["id"])]

    def scout_github(topic, kb, cfg_, *, papers, queries=None):
        rec.events.append(("scout_github", [p["id"] for p in papers], queries))
        return [repo("a/b")] if queries is None else []

    def read(p, topic, kb, cfg_):
        rec.events.append(("reader", p["id"]))
        failure = state["reader_fail"].get(p["id"])
        if failure:
            raise failure
        return dict(BD)

    def ideas_run(topic, kb, cfg_):
        rec.events.append(("ideas", (kb.root / "gap-map.md").exists()))
        return [{"title": "I1", "rank": 1, "novelty": "unclear"}]

    def critic_run(topic, ideas, kb, cfg_, *, final):
        rec.events.append(("critic", final))
        queries = state["critic"].pop(0) if state["critic"] else []
        return ideas, ([] if final else queries)

    monkeypatch.setattr(pipeline.scout_arxiv, "run", scout_arxiv)
    monkeypatch.setattr(pipeline.scout_github, "run", scout_github)
    monkeypatch.setattr(pipeline.reader, "run", read)
    monkeypatch.setattr(pipeline.ideas, "run", ideas_run)
    monkeypatch.setattr(pipeline.critic, "run", critic_run)
    monkeypatch.setattr(pipeline, "load_env", lambda: None)
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "check_cli", lambda: "claude")
    monkeypatch.setattr(pipeline, "init_langfuse", lambda: None)
    return rec, cfg, state


def _names(rec, name):
    return [e for e in rec.events if e[0] == name]


def test_run_order_and_caps(world):
    rec, cfg, _ = world
    result = pipeline.run("humanoid wbc", cfg)

    scouts = _names(rec, "scout_arxiv")
    assert scouts[0] == ("scout_arxiv", cfg["max_new_papers"] - cfg["followup_papers"], None)
    assert scouts[1] == ("scout_arxiv", cfg["followup_papers"], ["more q"])
    assert _names(rec, "scout_github")[1] == ("scout_github", ["F1"], ["more q"])
    assert _names(rec, "critic") == [("critic", False), ("critic", True)]
    assert _names(rec, "ideas") == [("ideas", True)]  # gap map written before ideas
    assert result.critic_rounds == 2 and result.papers_read == 3 and result.repos_found == 1
    assert result.ideas_path.exists() and result.stopped_early is None
    kb = KnowledgeBase(cfg, "humanoid wbc")
    assert all(kb.has_paper(i) for i in ("P1", "P2", "F1")) and kb.has_repo("a/b")


def test_run_skips_followup_when_no_queries(world):
    rec, cfg, state = world
    state["critic"] = [[]]
    result = pipeline.run("t", cfg)
    assert len(_names(rec, "scout_arxiv")) == 1
    assert _names(rec, "critic") == [("critic", False)]
    assert result.critic_rounds == 1


def test_reader_failure_skips_paper(world):
    rec, cfg, state = world
    state["reader_fail"] = {"P1": ReaderError("bad"), "F1": RuntimeError("network")}
    result = pipeline.run("t", cfg)
    kb = KnowledgeBase(cfg, "t")
    assert not kb.has_paper("P1") and not kb.has_paper("F1") and kb.has_paper("P2")
    assert result.papers_read == 1 and result.ideas_path.exists()


def test_usage_limit_stops_and_resumes(world, capsys):
    rec, cfg, state = world
    cfg["llm_concurrency"] = 1  # deterministic read order
    state["initial"] = [paper("P1"), paper("P2"), paper("P3")]
    state["reader_fail"] = {"P2": LLMUsageLimitError("Claude usage limit reached")}

    assert pipeline.main(["t"]) == 2
    kb = KnowledgeBase(cfg, "t")
    assert kb.has_paper("P1") and not kb.has_paper("P2")
    assert list((kb.root / "ideas").iterdir()) == []
    assert (kb.root / "gap-map.md").exists()

    rec.events.clear()
    state["reader_fail"] = {}
    pipeline.run("t", cfg)
    assert [e[1] for e in _names(rec, "reader")][:2] == ["P2", "P3"]
    assert ("reader", "P1") not in rec.events


def test_inbox_papers_read_uncapped_and_marked_done(world, monkeypatch):
    rec, cfg, state = world
    cfg["max_new_papers"], cfg["followup_papers"] = 1, 0
    state["initial"], state["critic"] = [], [[]]
    inbox = [paper(f"local-{i}", source="inbox", content="x", content_type="pdf_text") for i in range(3)]
    done = []
    monkeypatch.setattr(pipeline.KnowledgeBase, "ingest_inbox", lambda self: [dict(p) for p in inbox])
    monkeypatch.setattr(pipeline.KnowledgeBase, "mark_inbox_done", lambda self, p: done.append(p["id"]))

    result = pipeline.run("t", cfg)
    assert sorted(e[1] for e in _names(rec, "reader")) == ["local-0", "local-1", "local-2"]
    assert sorted(done) == ["local-0", "local-1", "local-2"]
    assert result.papers_read == 3


def test_main_requires_topic(world, capsys):
    assert pipeline.main([]) == 1
    assert 'python pipeline.py "<topic>"' in capsys.readouterr().out


def test_main_checks_cli_first(world, monkeypatch):
    rec, _, _ = world

    def missing():
        raise SystemExit("Claude Code (the `claude` command) was not found.")

    monkeypatch.setattr(pipeline, "check_cli", missing)
    with pytest.raises(SystemExit):
        pipeline.main(["t"])
    assert rec.events == []


def test_main_success_prints_summary(world, capsys):
    assert pipeline.main(["humanoid", "wbc"]) == 0
    out = capsys.readouterr().out
    assert "ideas" in out and "Papers read: 3" in out
