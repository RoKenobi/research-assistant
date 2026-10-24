import json

import pytest

from agents import reader
from agents.reader import ReaderError
from core.kb import KnowledgeBase
from fakes import FakeLLM, make_cfg, paper

GOOD = {
    "problem": "Stay balanced on stairs.", "approach": "PPO with a height map.",
    "setup": {"embodiment": "Unitree G1", "simulator": "Isaac Lab", "sim_or_real": "both",
              "training_cost": "1 GPU-day"},
    "results": "95% success over 200 trials.",
    "limitations": {"stated": ["no carrying"], "unstated": ["single gait"]},
    "future_work": ["loco-manipulation"], "code_and_data": [],
    "key_references": [{"title": "Prior", "arxiv_id": "2101.00001"}],
    "relevance": "Core WBC paper.", "approaches": ["  RL ", "rl", "Sim-to-Real"],
    "capabilities": ["Stairs"], "lab_fit": 7,
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    kb = KnowledgeBase(cfg, "humanoid wbc")
    fetched = []
    monkeypatch.setattr(reader.arxiv_source, "fetch_content",
                        lambda p, budget: fetched.append(budget) or p.update(content="FULL BODY TEXT",
                                                                             content_type="latex"))
    return cfg, kb, fetched


def test_reader_returns_breakdown_with_all_keys(setup, monkeypatch):
    cfg, kb, fetched = setup
    monkeypatch.setattr(reader, "complete", FakeLLM({"reader": "```json\n" + json.dumps(GOOD) + "\n```"}))
    bd = reader.run(paper("2401.00001"), "humanoid wbc", kb, cfg)
    assert fetched == [cfg["paper_char_budget"]]
    assert bd["approaches"] == ["rl", "sim-to-real"]
    assert bd["capabilities"] == ["stairs"]
    assert bd["lab_fit"] == 3
    assert bd["setup"]["embodiment"] == "Unitree G1"
    assert bd["limitations"]["unstated"] == ["single gait"]


def test_reader_fills_missing_keys(setup, monkeypatch):
    cfg, kb, _ = setup
    monkeypatch.setattr(reader, "complete", FakeLLM({"reader": '{"problem": "p", "lab_fit": "high"}'}))
    bd = reader.run(paper("2401.00001"), "t", kb, cfg)
    assert bd["problem"] == "p"
    assert bd["limitations"] == {"stated": [], "unstated": []}
    assert bd["setup"] == {"embodiment": "", "simulator": "", "sim_or_real": "", "training_cost": ""}
    assert bd["key_references"] == [] and bd["approaches"] == [] and bd["lab_fit"] == 0


def test_reader_prompt_has_content_lab_and_known_tags(setup, monkeypatch):
    cfg, kb, _ = setup
    kb.add_paper(paper("2301.00001"), {"approaches": ["mpc"], "capabilities": ["push recovery"]})
    llm = FakeLLM({"reader": json.dumps(GOOD)})
    monkeypatch.setattr(reader, "complete", llm)
    reader.run(paper("2401.00001"), "humanoid wbc", kb, cfg)

    call = llm.calls[0]
    for text in ("FULL BODY TEXT", "Unitree G1", "AgiBot X2", "MuJoCo", "still relevant",
                 "mpc", "push recovery", "Reuse"):
        assert text in call["prompt"], text
    assert call["model"] == cfg["models"]["deep"] and call["effort"] == cfg["effort"]["deep"]


def test_reader_retries_once_then_raises(setup, monkeypatch):
    cfg, kb, _ = setup
    llm = FakeLLM({"reader": ["not json", "still not json"]})
    monkeypatch.setattr(reader, "complete", llm)
    with pytest.raises(ReaderError):
        reader.run(paper("2401.00001"), "t", kb, cfg)
    assert len(llm.calls) == 2

    llm = FakeLLM({"reader": ["oops", json.dumps(GOOD)]})
    monkeypatch.setattr(reader, "complete", llm)
    assert reader.run(paper("2401.00001"), "t", kb, cfg)["problem"] == GOOD["problem"]


def test_reader_abstract_only_paper(setup, monkeypatch):
    cfg, kb, _ = setup
    monkeypatch.setattr(reader.arxiv_source, "fetch_content",
                        lambda p, budget: p.update(content=p["abstract"], content_type="abstract"))
    llm = FakeLLM({"reader": json.dumps(GOOD)})
    monkeypatch.setattr(reader, "complete", llm)
    bd = reader.run(paper("2401.00001"), "t", kb, cfg)
    assert "only the abstract" in llm.calls[0]["prompt"].lower()
    assert bd["results"] == GOOD["results"]


def test_reader_uses_existing_content(setup, monkeypatch):
    cfg, kb, fetched = setup
    llm = FakeLLM({"reader": json.dumps(GOOD)})
    monkeypatch.setattr(reader, "complete", llm)
    reader.run(paper("local-notes", content="MY PDF TEXT", content_type="pdf_text"), "t", kb, cfg)
    assert fetched == [] and "MY PDF TEXT" in llm.calls[0]["prompt"]
