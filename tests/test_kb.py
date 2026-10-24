import json
from datetime import date

import pymupdf
import pytest
import yaml

from core import kb as kb_module
from core.kb import KnowledgeBase, slugify


def _cfg(tmp_path):
    return {"_vault": tmp_path, "research_folder": "Research", "paper_char_budget": 40000}


def _paper(pid="2307.15818", **kw):
    paper = {"id": pid, "title": f"Paper {pid}", "authors": ["A. Author"], "abstract": "abs",
             "pdf_url": "", "published": "2023-07-28", "source": "arxiv",
             "content": "body", "content_type": "latex"}
    paper.update(kw)
    return paper


def _breakdown(approaches=("rl",), capabilities=("stairs",), **kw):
    bd = {
        "problem": "Climb stairs while pushed.", "approach": "PPO policy with a terrain map.",
        "setup": {"embodiment": "Unitree G1", "simulator": "MuJoCo", "sim_or_real": "both",
                  "training_cost": "8 GPU-hours"},
        "results": "92% success.", "limitations": {"stated": ["flat stairs only"], "unstated": ["no load"]},
        "future_work": ["carry objects"], "code_and_data": ["https://github.com/a/b"],
        "key_references": [{"title": "Prior work", "arxiv_id": "2101.00001"}, {"title": "Book", "arxiv_id": None}],
        "relevance": "Directly about WBC on stairs.", "approaches": list(approaches),
        "capabilities": list(capabilities), "lab_fit": 3,
    }
    bd.update(kw)
    return bd


def _frontmatter(text):
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm), body


def test_slugify_same_topic_different_spelling():
    assert slugify("Humanoid WBC") == "humanoid-wbc"
    assert slugify("humanoid  wbc") == "humanoid-wbc"
    assert slugify(" Humanoid-WBC! ") == "humanoid-wbc"
    assert slugify("!!!") == "untitled"


def test_kb_creates_topic_folders(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "Humanoid WBC")
    assert kb.root == tmp_path / "Research" / "humanoid-wbc"
    for sub in ("papers", "repos", "ideas", "inbox", "inbox/processed"):
        assert (kb.root / sub).is_dir()


def test_add_paper_writes_note_and_index(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "Humanoid WBC")
    path = kb.add_paper(_paper(), _breakdown(approaches=["rl"], capabilities=["push recovery"]))

    assert path == kb.root / "papers" / "2307.15818.md"
    fm, body = _frontmatter(path.read_text(encoding="utf-8"))
    assert fm["arxiv_id"] == "2307.15818" and fm["title"] == "Paper 2307.15818"
    assert {"humanoid-wbc", "rl", "push-recovery"} <= set(fm["tags"])
    for heading in ("## Problem", "## Approach", "## Setup", "## Results", "## Limitations",
                    "## Future work", "## Code & data", "## Key references", "## Relevance"):
        assert heading in body
    assert "https://arxiv.org/abs/2307.15818" in body
    assert "no load" in body and "Unitree G1" in body

    reloaded = KnowledgeBase(_cfg(tmp_path), "humanoid wbc")
    assert reloaded.has_paper("2307.15818") and not reloaded.has_paper("9999.00000")
    digest = reloaded.digest()[0]
    assert digest["id"] == "2307.15818" and digest["problem"] == "Climb stairs while pushed."
    assert digest["limitations"] == {"stated": ["flat stairs only"], "unstated": ["no load"]}


def test_local_paper_has_no_arxiv_link(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    path = kb.add_paper(_paper("local-my-notes", source="inbox", content_type="pdf_text"), _breakdown())
    fm, body = _frontmatter(path.read_text(encoding="utf-8"))
    assert fm.get("arxiv_id") is None
    assert "arxiv.org/abs/local" not in body


def test_known_tags_sorted_and_unique(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    kb.add_paper(_paper("A"), _breakdown(approaches=["rl", "mpc"], capabilities=["stairs"]))
    kb.add_paper(_paper("B"), _breakdown(approaches=["mpc"], capabilities=["push recovery"]))
    assert kb.known_tags() == {"approaches": ["mpc", "rl"], "capabilities": ["push recovery", "stairs"]}


def test_gap_map_marks_empty_cells(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    kb.add_paper(_paper("A"), _breakdown(approaches=["rl"], capabilities=["stairs"]))
    kb.add_paper(_paper("B"), _breakdown(approaches=["mpc"], capabilities=["push recovery"]))
    text = kb.write_gap_map().read_text(encoding="utf-8")

    rows = {line.split("|")[1].strip(): line for line in text.splitlines()
            if line.startswith("| ") and "---" not in line}
    header = next(line for line in text.splitlines() if line.startswith("| Approach"))
    columns = [c.strip() for c in header.strip("|").split("|")][1:]
    assert columns == ["push recovery", "stairs"]
    # Obsidian needs the alias pipe escaped inside a table cell.
    rl_cells = [c.strip() for c in rows["rl"].split(" | ")]
    assert rl_cells[1] == "—"
    assert "[[papers/A\\|A]]" in rl_cells[2]
    assert set(rows) >= {"rl", "mpc"}


def test_ingest_inbox_pdf_and_links(tmp_path, monkeypatch):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Whole-body control on stairs, my own notes.")
    doc.save(kb.root / "inbox" / "My Notes.pdf")
    doc.close()
    (kb.root / "inbox" / "links.txt").write_text(
        "https://arxiv.org/abs/2307.15818v2\nsome comment\n", encoding="utf-8")

    looked_up = []

    def fake_fetch(ids, source="arxiv"):
        looked_up.append((ids, source))
        return [_paper(i, source=source, content=None, content_type=None) for i in ids]

    monkeypatch.setattr(kb_module.arxiv_source, "fetch_by_ids", fake_fetch)
    papers = kb.ingest_inbox()

    assert looked_up == [(["2307.15818"], "inbox")]
    by_id = {p["id"]: p for p in papers}
    assert set(by_id) == {"local-my-notes", "2307.15818"}
    assert all(p["source"] == "inbox" for p in papers)
    local = by_id["local-my-notes"]
    assert "stairs" in local["content"] and local["content_type"] == "pdf_text"
    assert local["title"] == "My Notes"

    for paper in papers:
        kb.mark_inbox_done(paper)
    assert (kb.root / "inbox" / "processed" / "My Notes.pdf").exists()
    assert not (kb.root / "inbox" / "My Notes.pdf").exists()
    assert "2307.15818" not in (kb.root / "inbox" / "links.txt").read_text(encoding="utf-8")
    assert "some comment" in (kb.root / "inbox" / "links.txt").read_text(encoding="utf-8")
    assert "2307.15818" in (kb.root / "inbox" / "processed" / "links.txt").read_text(encoding="utf-8")


def test_ingest_inbox_skips_known_papers(tmp_path, monkeypatch):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    kb.add_paper(_paper("2307.15818"), _breakdown())
    (kb.root / "inbox" / "links.txt").write_text("2307.15818\n", encoding="utf-8")
    monkeypatch.setattr(kb_module.arxiv_source, "fetch_by_ids",
                        lambda ids, source="arxiv": pytest.fail("known id looked up"))
    assert kb.ingest_inbox() == []


def test_write_ideas_does_not_overwrite_same_day(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    kb.add_paper(_paper("A"), _breakdown())
    idea = {"title": "Stairs with load", "improvement": "Add payload estimation.", "gap": "rl × carrying",
            "evidence_papers": ["A"], "closest_prior_work": ["A"], "novelty": "novel",
            "novelty_evidence": "No paper combines both.", "feasibility": "Sim-first in MuJoCo on G1.",
            "rank": 1}
    meta = {"papers_read": 1, "repos_found": 0, "critic_rounds": 1, "stopped_early": None}
    first = kb.write_ideas([idea], meta)
    second = kb.write_ideas([{**idea, "title": "Second batch"}], meta)

    today = date.today().isoformat()
    assert first.name == f"{today}.md" and second.name == f"{today}-2.md"
    text = first.read_text(encoding="utf-8")
    assert "## 1. Stairs with load" in text and "**novel**" in text and "[[papers/A]]" in text
    assert kb.last_ideas()[0]["title"] == "Second batch"


def test_index_survives_reload_and_is_atomic(tmp_path):
    kb = KnowledgeBase(_cfg(tmp_path), "t")
    repo = {"full_name": "a/b", "url": "https://github.com/a/b", "description": "WBC code", "stars": 5,
            "updated": "2026-09-01", "language": "Python", "linked_papers": ["A"], "why_relevant": "G1 WBC."}
    path = kb.add_repo(repo)
    assert path == kb.root / "repos" / "a__b.md"
    assert "[[papers/A]]" in path.read_text(encoding="utf-8")
    index = json.loads((kb.root / "index.json").read_text(encoding="utf-8"))
    assert index["repos"]["a/b"]["stars"] == 5
    assert not (kb.root / "index.json.tmp").exists()
    reloaded = KnowledgeBase(_cfg(tmp_path), "t")
    assert reloaded.has_repo("a/b")
    assert reloaded.repos_digest() == [{"full_name": "a/b", "description": "WBC code", "stars": 5,
                                        "linked_papers": ["A"], "why_relevant": "G1 WBC."}]
