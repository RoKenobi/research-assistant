import json

import pytest

from agents import critic, ideas
from core.kb import KnowledgeBase
from fakes import FakeLLM, make_cfg, paper, repo


@pytest.fixture
def kb(tmp_path):
    cfg = make_cfg(tmp_path)
    kb = KnowledgeBase(cfg, "humanoid wbc")
    kb.add_paper(paper("A", content="SENTINEL_FULL_TEXT"),
                 {"problem": "Balance under pushes.", "approaches": ["rl"], "capabilities": ["push recovery"]})
    kb.add_repo(repo("lab/g1-wbc"))
    kb.write_gap_map()
    kb.write_ideas([{"title": "Old idea", "novelty": "done", "rank": 1}], {})
    return kb


def _idea(title, rank, **kw):
    base = {"title": title, "improvement": "i", "gap": "g", "evidence_papers": ["A"],
            "closest_prior_work": [], "feasibility": "f", "rank": rank}
    base.update(kw)
    return base


def test_ideas_prompt_contains_digest_gap_map_repos_and_last_ideas(kb, monkeypatch):
    llm = FakeLLM({"ideas": json.dumps([_idea("New", 1)])})
    monkeypatch.setattr(ideas, "complete", llm)
    ideas.run("humanoid wbc", kb, kb.cfg)

    call = llm.calls[0]
    prompt = call["prompt"]
    assert "Balance under pushes." in prompt
    assert "| Approach \ Capability |" in prompt
    assert "lab/g1-wbc" in prompt
    assert "Old idea" in prompt and "already proposed" in prompt.lower()
    assert "SENTINEL_FULL_TEXT" not in prompt
    assert call["model"] == kb.cfg["models"]["deep"]


def test_ideas_normalizes_and_ranks(kb, monkeypatch):
    reply = "Ideas:\n" + json.dumps([{"title": "Second", "rank": 3}, _idea("First", 1, novelty="novel")])
    monkeypatch.setattr(ideas, "complete", FakeLLM({"ideas": reply}))
    got = ideas.run("t", kb, kb.cfg)
    assert [i["title"] for i in got] == ["First", "Second"]
    assert [i["rank"] for i in got] == [1, 2]
    assert all(i["novelty"] == "unclear" for i in got)
    assert got[1]["evidence_papers"] == [] and got[1]["improvement"] == ""


def test_ideas_retries_then_returns_empty(kb, monkeypatch):
    llm = FakeLLM({"ideas": ["junk", "more junk"]})
    monkeypatch.setattr(ideas, "complete", llm)
    assert ideas.run("t", kb, kb.cfg) == []
    assert len(llm.calls) == 2


def _verdicts(*items):
    return json.dumps(list(items))


def test_critic_returns_followups_only_for_unclear(kb, monkeypatch):
    current = [_idea("Load-aware stairs", 1, novelty="unclear"), _idea("Push recovery on X2", 2, novelty="unclear")]
    reply = _verdicts(
        {"title": "load-aware stairs ", "novelty": "unclear", "novelty_evidence": "not sure",
         "feasibility": "MuJoCo first", "rank": 2, "search_queries": ["payload stairs humanoid", "b"]},
        {"title": "Push recovery on X2", "novelty": "novel", "novelty_evidence": "none found",
         "feasibility": "needs X2", "rank": 1, "search_queries": ["ignored"]},
    )
    llm = FakeLLM({"critic": [reply, reply]})
    monkeypatch.setattr(critic, "complete", llm)

    got, queries = critic.run("t", current, kb, kb.cfg, final=False)
    assert queries == ["payload stairs humanoid", "b"]
    assert [i["title"] for i in got] == ["Push recovery on X2", "Load-aware stairs"]
    assert got[0]["novelty"] == "novel" and got[0]["feasibility"] == "needs X2"
    assert "search_queries" not in got[0]

    _, queries = critic.run("t", current, kb, kb.cfg, final=True)
    assert queries == []
    assert "final" in llm.calls[1]["prompt"].lower()


def test_critic_drops_unknown_verdicts_and_keeps_omitted(kb, monkeypatch):
    current = [_idea("One", 1, novelty="unclear", feasibility="orig"), _idea("Two", 2, novelty="unclear")]
    reply = _verdicts({"title": "One", "novelty": "maybe", "rank": 1})
    monkeypatch.setattr(critic, "complete", FakeLLM({"critic": reply}))
    got, _ = critic.run("t", current, kb, kb.cfg, final=False)
    assert got[0]["novelty"] == "unclear" and got[0]["feasibility"] == "orig"
    assert got[1]["title"] == "Two"


def test_critic_caps_followup_queries(kb, monkeypatch):
    current = [_idea("One", 1)]
    reply = _verdicts({"title": "One", "novelty": "unclear", "search_queries": ["a", "b", "c", "d", "e", "a"]})
    monkeypatch.setattr(critic, "complete", FakeLLM({"critic": reply}))
    _, queries = critic.run("t", current, kb, kb.cfg, final=False)
    assert queries == ["a", "b", "c", "d"]


def test_critic_unparseable_reply_keeps_ideas(kb, monkeypatch):
    current = [_idea("One", 1, novelty="unclear")]
    monkeypatch.setattr(critic, "complete", FakeLLM({"critic": "no json"}))
    got, queries = critic.run("t", current, kb, kb.cfg, final=False)
    assert got == current and queries == []
