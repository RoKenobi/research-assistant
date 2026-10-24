"""Per-topic knowledge base in the Obsidian vault.

<vault>/<research_folder>/<topic-slug>/
    papers/<id>.md, repos/<owner>__<repo>.md, gap-map.md, ideas/YYYY-MM-DD.md,
    inbox/ (+ processed/), index.json

index.json is the machine-readable side: what has been read (for dedupe and
resume) and a compact digest per paper that the Ideas and Critic agents read
instead of the full notes.
"""
import json
import logging
import os
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

from sources import arxiv_source

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_DIGEST_KEYS = ("problem", "approach", "setup", "limitations", "future_work", "relevance",
                "approaches", "capabilities", "lab_fit")


def slugify(topic: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", topic.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "untitled"


def _tag(text: str) -> str:
    """Obsidian tags can't contain spaces."""
    return re.sub(r"\s+", "-", text.strip())


def _is_arxiv(paper_id: str) -> bool:
    return bool(_ARXIV_ID_RE.fullmatch(paper_id))


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def _frontmatter(data: dict) -> str:
    return f"---\n{yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)}---\n\n"


class KnowledgeBase:
    def __init__(self, cfg: dict, topic: str):
        self.cfg = cfg
        self.topic = topic
        self.slug = slugify(topic)
        self.root: Path = cfg["_vault"] / cfg["research_folder"] / self.slug
        for sub in ("papers", "repos", "ideas", "inbox", "inbox/processed"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        if self._index_path.exists():
            self.index = json.loads(self._index_path.read_text(encoding="utf-8"))
        else:
            self.index = {"topic": topic, "slug": self.slug, "papers": {}, "repos": {}, "ideas_history": []}

    # --- index -----------------------------------------------------------------

    def save(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._index_path)  # never leaves a half-written index behind

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.index["papers"]

    def has_repo(self, full_name: str) -> bool:
        return full_name in self.index["repos"]

    def digest(self) -> list[dict]:
        # Snapshot: reader threads call this while the main thread adds papers.
        return [entry["digest"] for entry in list(self.index["papers"].values())]

    def repos_digest(self) -> list[dict]:
        keys = ("full_name", "description", "stars", "linked_papers", "why_relevant")
        return [{k: repo.get(k) for k in keys} for repo in self.index["repos"].values()]

    def known_tags(self) -> dict[str, list[str]]:
        approaches, capabilities = set(), set()
        for d in self.digest():
            approaches.update(d.get("approaches") or [])
            capabilities.update(d.get("capabilities") or [])
        return {"approaches": sorted(approaches), "capabilities": sorted(capabilities)}

    def last_ideas(self) -> list[dict]:
        history = self.index["ideas_history"]
        return history[-1]["ideas"] if history else []

    # --- papers and repos -----------------------------------------------------

    def add_paper(self, paper: dict, breakdown: dict) -> Path:
        pid = paper["id"]
        path = self.root / "papers" / f"{pid}.md"
        path.write_text(self._paper_note(paper, breakdown), encoding="utf-8")
        self.index["papers"][pid] = {
            "title": paper["title"],
            "note": f"papers/{pid}.md",
            "added": date.today().isoformat(),
            "source": paper["source"],
            "digest": {"id": pid, "title": paper["title"], **{k: breakdown.get(k) for k in _DIGEST_KEYS}},
        }
        self.save()
        return path

    def _paper_note(self, paper: dict, bd: dict) -> str:
        pid = paper["id"]
        arxiv = _is_arxiv(pid)
        fm = {
            "arxiv_id": pid if arxiv else None,
            "id": pid,
            "title": paper["title"],
            "authors": paper.get("authors") or [],
            "published": paper.get("published") or "",
            "source": paper["source"],
            "read_from": paper.get("content_type"),
            "lab_fit": bd.get("lab_fit"),
            "tags": ["research", self.slug, *(_tag(t) for t in bd.get("approaches", []) + bd.get("capabilities", []))],
        }
        setup = bd.get("setup") or {}
        lim = bd.get("limitations") or {}
        refs = []
        for ref in bd.get("key_references") or []:
            rid = ref.get("arxiv_id")
            if rid and self.has_paper(rid):
                refs.append(f"- [[papers/{rid}|{ref.get('title', rid)}]]")
            elif rid:
                refs.append(f"- {ref.get('title', rid)} ([arXiv](https://arxiv.org/abs/{rid}))")
            else:
                refs.append(f"- {ref.get('title', '')}")
        byline = [", ".join(paper.get("authors") or []), paper.get("published") or ""]
        if arxiv:
            byline.insert(0, f"[arXiv {pid}](https://arxiv.org/abs/{pid})")
        if paper.get("content_type") == "abstract":
            byline.append("read from the abstract only")
        return (
            _frontmatter(fm)
            + f"# {paper['title']}\n\n{' · '.join(b for b in byline if b)}\n\n"
            + f"## Problem\n{bd.get('problem', '')}\n\n"
            + f"## Approach\n{bd.get('approach', '')}\n\n"
            + "## Setup\n"
            + f"- **Embodiment:** {setup.get('embodiment', '')}\n- **Simulator:** {setup.get('simulator', '')}\n"
            + f"- **Sim or real:** {setup.get('sim_or_real', '')}\n- **Training cost:** {setup.get('training_cost', '')}\n\n"
            + f"## Results\n{bd.get('results', '')}\n\n"
            + f"## Limitations\n**Stated**\n{_bullets(lim.get('stated') or [])}\n\n"
            + f"**Unstated**\n{_bullets(lim.get('unstated') or [])}\n\n"
            + f"## Future work\n{_bullets(bd.get('future_work') or [])}\n\n"
            + f"## Code & data\n{_bullets(bd.get('code_and_data') or [])}\n\n"
            + f"## Key references\n{chr(10).join(refs) or '- (none)'}\n\n"
            + f"## Relevance\n{bd.get('relevance', '')}\n\nLab fit: {bd.get('lab_fit', 0)}/3\n"
        )

    def add_repo(self, repo: dict) -> Path:
        path = self.root / "repos" / f"{repo['full_name'].replace('/', '__')}.md"
        linked = "\n".join(f"- [[papers/{pid}]]" for pid in repo.get("linked_papers") or []) or "- (none)"
        path.write_text(
            _frontmatter({"repo": repo["full_name"], "url": repo["url"], "stars": repo.get("stars", 0),
                          "updated": repo.get("updated", ""), "language": repo.get("language", ""),
                          "tags": ["research", self.slug, "repo"]})
            + f"# {repo['full_name']}\n\n[{repo['url']}]({repo['url']}) · ★ {repo.get('stars', 0)}"
            + f" · updated {repo.get('updated', '')}\n\n{repo.get('description', '')}\n\n"
            + f"## Why it matters\n{repo.get('why_relevant', '')}\n\n## Linked papers\n{linked}\n",
            encoding="utf-8",
        )
        self.index["repos"][repo["full_name"]] = repo
        self.save()
        return path

    # --- gap map and ideas ----------------------------------------------------

    def write_gap_map(self) -> Path:
        tags = self.known_tags()
        rows, cols = tags["approaches"], tags["capabilities"]
        lines = [_frontmatter({"topic": self.topic, "tags": ["research", self.slug, "gap-map"]}),
                 f"# Gap map: {self.topic}\n",
                 f"{len(self.index['papers'])} papers. Rows are approaches, columns are capabilities. "
                 "A “—” cell is a combination no paper in this knowledge base covers yet.\n"]
        if not rows or not cols:
            lines.append("No papers yet.\n")
        else:
            cells: dict[tuple[str, str], list[str]] = {}
            for d in self.digest():
                for a in d.get("approaches") or []:
                    for c in d.get("capabilities") or []:
                        cells.setdefault((a, c), []).append(d["id"])
            lines.append("| Approach \\ Capability | " + " | ".join(cols) + " |")
            lines.append("|" + " --- |" * (len(cols) + 1))
            for a in rows:
                # The alias pipe must be escaped inside a table cell or Obsidian splits the cell.
                row = [", ".join(f"[[papers/{pid}\\|{pid}]]" for pid in cells.get((a, c), [])) or "—"
                       for c in cols]
                lines.append(f"| {a} | " + " | ".join(row) + " |")
        path = self.root / "gap-map.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_ideas(self, ideas: list[dict], meta: dict) -> Path:
        today = date.today().isoformat()
        path, n = self.root / "ideas" / f"{today}.md", 1
        while path.exists():
            n += 1
            path = self.root / "ideas" / f"{today}-{n}.md"

        summary = (f"Papers read this run: {meta.get('papers_read', 0)} · repos found: "
                   f"{meta.get('repos_found', 0)} · critic rounds: {meta.get('critic_rounds', 0)} · "
                   f"knowledge base: {len(self.index['papers'])} papers")
        if meta.get("stopped_early"):
            summary += f"\n\n> [!WARNING]\n> Stopped early: {meta['stopped_early']}"
        parts = [_frontmatter({"date": today, "topic": self.topic, "tags": ["research", self.slug, "ideas"]}),
                 f"# Research ideas: {self.topic}\n\n{summary}\n\nSee also [[gap-map]].\n"]
        for idea in sorted(ideas, key=lambda i: i.get("rank", 99)):
            evidence = ", ".join(f"[[papers/{pid}]]" for pid in idea.get("evidence_papers") or []) or "—"
            parts.append(
                f"## {idea.get('rank', '?')}. {idea.get('title', '')}\n\n{idea.get('improvement', '')}\n\n"
                f"- **Gap it fills:** {idea.get('gap', '')}\n"
                f"- **Novelty:** **{idea.get('novelty', 'unclear')}**. {idea.get('novelty_evidence', '')}\n"
                f"- **Closest prior work:** {', '.join(idea.get('closest_prior_work') or []) or '—'}\n"
                f"- **Feasibility:** {idea.get('feasibility', '')}\n"
                f"- **Evidence:** {evidence}\n"
            )
        path.write_text("\n".join(parts), encoding="utf-8")
        self.index["ideas_history"].append({"date": today, "file": f"ideas/{path.name}", "ideas": ideas})
        self.save()
        return path

    # --- inbox ------------------------------------------------------------------

    def ingest_inbox(self) -> list[dict]:
        """The user's own papers: PDFs and arXiv links dropped in inbox/."""
        inbox = self.root / "inbox"
        papers = []
        for pdf in sorted(inbox.glob("*.pdf")):
            pid = f"local-{slugify(pdf.stem)}"
            if self.has_paper(pid):
                continue
            text = arxiv_source.pdf_text_from_bytes(pdf.read_bytes(), self.cfg["paper_char_budget"])
            papers.append({"id": pid, "title": pdf.stem, "authors": [], "abstract": text[:1500],
                           "pdf_url": "", "published": "", "source": "inbox",
                           "content": text or None, "content_type": "pdf_text" if text else None,
                           "_inbox_file": str(pdf)})

        links = inbox / "links.txt"
        if links.exists():
            line_for: dict[str, str] = {}
            for line in links.read_text(encoding="utf-8").splitlines():
                match = _ARXIV_ID_RE.search(line)
                if match and not self.has_paper(match.group(1)):
                    line_for.setdefault(match.group(1), line)
            for paper in arxiv_source.fetch_by_ids(list(line_for), source="inbox") if line_for else []:
                paper["_inbox_line"] = line_for.get(paper["id"], paper["id"])
                papers.append(paper)
        return papers

    def mark_inbox_done(self, paper: dict) -> None:
        processed = self.root / "inbox" / "processed"
        if paper.get("_inbox_file"):
            src = Path(paper["_inbox_file"])
            if src.exists():
                shutil.move(str(src), processed / src.name)
        if paper.get("_inbox_line"):
            links = self.root / "inbox" / "links.txt"
            if links.exists():
                kept = [l for l in links.read_text(encoding="utf-8").splitlines() if l != paper["_inbox_line"]]
                links.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            with open(processed / "links.txt", "a", encoding="utf-8") as f:
                f.write(paper["_inbox_line"] + "\n")
