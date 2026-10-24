"""Critic: has each idea been done already, and is it feasible for the lab?

When novelty is unclear it asks for targeted searches; the pipeline runs them
and calls the Critic again, at most max_critic_rounds times.
"""
import logging

from agents._shared import as_json, lab_context, model_args
from agents.ideas import NOVELTY, rerank
from core.jsonparse import parse_json
from core.kb import KnowledgeBase
from core.llm import complete
from core.tracing import observe, update_span

logger = logging.getLogger(__name__)

MAX_QUERIES = 4

SYSTEM = """\
You are a skeptical program-committee reviewer in embodied AI and robotics. For each proposed \
idea, judge whether it has already been done, using the papers provided. "done" and \
"partly done" must cite the paper ids that did it. Use "unclear" only when the provided papers \
can't settle it. Judge feasibility for the user's lab. Reply with JSON only."""

PROMPT = """\
Topic: {topic}
{lab}

Papers in the knowledge base (compact digest):
{digest}

Ideas to review:
{ideas}

{round_note}

Reply with a JSON list, one object per idea, matched by title:
{{"title": "same title as given", "novelty": "novel | partly done | done | unclear",
  "novelty_evidence": "why, citing paper ids", "feasibility": "fit for the lab's robots and simulator, sim-to-real risk, rough effort",
  "rank": 1{queries_field}}}
Re-rank so rank 1 is the idea most worth pursuing: novel or partly done, feasible, high impact."""

SEARCH_NOTE = ("If an idea's novelty is unclear, add up to 2 arXiv search queries (3-5 keywords each) that would settle it. "
               "Those searches will be run and you will review again.")
FINAL_NOTE = "This is the final review: no more searching. Give your best verdict for every idea."


@observe(name="critic")
def run(topic: str, ideas: list[dict], kb: KnowledgeBase, cfg: dict, *, final: bool) -> tuple[list[dict], list[str]]:
    prompt = PROMPT.format(
        topic=topic, lab=lab_context(cfg), digest=as_json(kb.digest()), ideas=as_json(ideas),
        round_note=FINAL_NOTE if final else SEARCH_NOTE,
        queries_field="" if final else ', "search_queries": ["only when novelty is unclear"]',
    )
    reply = complete(prompt, system=SYSTEM, name="critic", **model_args(cfg, "deep"))
    verdicts = [v for v in parse_json(reply.text, list) or [] if isinstance(v, dict)]
    if not verdicts:
        logger.warning("Critic: reply had no verdicts; keeping ideas as they are")
        return ideas, []

    by_title = {_key(v.get("title")): v for v in verdicts}
    reviewed, queries = [], []
    for idea in ideas:
        verdict = by_title.get(_key(idea["title"]))
        if verdict is None:
            reviewed.append(dict(idea))
            continue
        updated = dict(idea)
        updated["novelty"] = verdict.get("novelty") if verdict.get("novelty") in NOVELTY else "unclear"
        for key in ("novelty_evidence", "feasibility"):
            if verdict.get(key):
                updated[key] = str(verdict[key])
        try:
            updated["rank"] = int(verdict["rank"])
        except (KeyError, TypeError, ValueError):
            pass
        if not final and updated["novelty"] == "unclear":
            queries += [q.strip() for q in verdict.get("search_queries") or [] if isinstance(q, str) and q.strip()]
        reviewed.append(updated)

    queries = list(dict.fromkeys(queries))[:MAX_QUERIES]
    result = rerank(reviewed)
    update_span(input={"ideas": len(ideas), "final": final},
                metadata={"verdicts": {i["title"]: i["novelty"] for i in result}, "followup_queries": queries})
    return result, queries


def _key(title) -> str:
    return str(title or "").strip().lower()
