"""Reader: one paper -> the 9-point breakdown, plus tags for the gap map."""
import logging

from agents._shared import as_json, lab_context, model_args
from core.jsonparse import parse_json
from core.kb import KnowledgeBase
from core.llm import complete
from core.tracing import observe, update_span
from sources import arxiv_source

logger = logging.getLogger(__name__)

ATTEMPTS = 2

SYSTEM = """\
You are a PhD-level reviewer in embodied AI and robotics, reading a paper for a researcher \
who is looking for what to improve next. Be concrete and specific; quote numbers. Reply with \
one JSON object only, with exactly these keys:

{
  "problem": "one sentence: what exactly the paper tries to solve",
  "approach": "the method in plain words (RL / MPC / imitation / VLA / hybrid / learned model ...) and the key idea that makes it work",
  "setup": {"embodiment": "robot or agent", "simulator": "", "sim_or_real": "sim only | real only | both", "training_cost": ""},
  "results": "main numbers and what they were compared against",
  "limitations": {"stated": ["limits the authors admit"], "unstated": ["limits they don't mention: flat ground only, sim only, single robot, no payload ..."]},
  "future_work": ["what the authors say should come next"],
  "code_and_data": ["released code, models or datasets, with links"],
  "key_references": [{"title": "", "arxiv_id": "2101.00001 or null"}],
  "relevance": "how this connects to improving methods in the topic, and one idea it sparks",
  "approaches": ["short lowercase method tags"],
  "capabilities": ["short lowercase capability tags, e.g. stairs, push recovery, uneven terrain, loco-manipulation"],
  "lab_fit": 0
}

key_references: the 3-5 papers it builds on most. lab_fit: 0-3, how directly the work \
transfers to the user's lab setup. Use an empty string or list when the paper doesn't say."""

PROMPT = """\
Topic: {topic}
{lab}

Tags already used in this knowledge base. Reuse them when they fit, so the gap map lines up; \
only add a new tag for something genuinely different.
{tags}

Paper: {title}
Authors: {authors}
{content_note}

---
{content}"""

_EMPTY = {
    "problem": "", "approach": "", "results": "", "relevance": "",
    "future_work": [], "code_and_data": [], "key_references": [], "approaches": [], "capabilities": [],
}
_SETUP_KEYS = ("embodiment", "simulator", "sim_or_real", "training_cost")


class ReaderError(RuntimeError):
    pass


@observe(name="reader")
def run(paper: dict, topic: str, kb: KnowledgeBase, cfg: dict) -> dict:
    if paper["content"] is None:
        arxiv_source.fetch_content(paper, cfg["paper_char_budget"])

    note = {
        "latex": "Full text from the LaTeX source.",
        "pdf_text": "Full text extracted from the PDF.",
    }.get(paper["content_type"], "Only the abstract was available: say so where a field can't be judged.")
    prompt = PROMPT.format(
        topic=topic, lab=lab_context(cfg), tags=as_json(kb.known_tags()), title=paper["title"],
        authors=", ".join(paper.get("authors") or []), content_note=note, content=paper["content"] or "",
    )

    for attempt in range(1, ATTEMPTS + 1):
        reply = complete(prompt, system=SYSTEM, name="reader", **model_args(cfg, "deep"))
        parsed = parse_json(reply.text, dict)
        if parsed is not None:
            breakdown = normalize(parsed)
            update_span(input={"paper": paper["id"], "content_type": paper["content_type"]},
                        metadata={"attempts": attempt, "approaches": breakdown["approaches"],
                                  "capabilities": breakdown["capabilities"]})
            return breakdown
        logger.warning("Reader: reply for %s was not JSON (attempt %d)", paper["id"], attempt)
    raise ReaderError(f"No usable breakdown for {paper['id']} after {ATTEMPTS} attempts")


def normalize(raw: dict) -> dict:
    """Fill missing keys with empty values of the right type and tidy the tags."""
    bd = {k: raw.get(k) if isinstance(raw.get(k), type(v)) else v for k, v in _EMPTY.items()}
    setup = raw.get("setup") if isinstance(raw.get("setup"), dict) else {}
    bd["setup"] = {k: str(setup.get(k) or "") for k in _SETUP_KEYS}
    lim = raw.get("limitations") if isinstance(raw.get("limitations"), dict) else {}
    bd["limitations"] = {k: [str(x) for x in lim.get(k) or [] if x] for k in ("stated", "unstated")}
    for key in ("approaches", "capabilities"):
        bd[key] = list(dict.fromkeys(t.strip().lower() for t in bd[key] if isinstance(t, str) and t.strip()))
    bd["key_references"] = [
        {"title": str(r.get("title") or ""), "arxiv_id": r.get("arxiv_id") or None}
        for r in bd["key_references"] if isinstance(r, dict)
    ]
    try:
        bd["lab_fit"] = max(0, min(3, int(raw.get("lab_fit", 0))))
    except (TypeError, ValueError):
        bd["lab_fit"] = 0
    return bd
