"""Ideas: the topic's whole knowledge base -> several ranked research ideas."""
import logging

from agents._shared import as_json, lab_context, model_args
from core.jsonparse import parse_json
from core.kb import KnowledgeBase
from core.llm import complete
from core.tracing import observe, update_span

logger = logging.getLogger(__name__)

ATTEMPTS = 2
NOVELTY = ("novel", "partly done", "done", "unclear")

SYSTEM = """\
You are a senior embodied-AI researcher helping a new PhD student find a paper to write. \
Propose several concrete improvements to existing methods or systems in the topic, not one \
grand idea. Every idea must be grounded: tie it to an empty cell of the gap map or to a stated \
or unstated limitation, and cite the paper ids that show the gap. Prefer ideas a small lab can \
test in simulation first. Reply with JSON only."""

PROMPT = """\
Topic: {topic}
{lab}

Gap map (rows = approaches, columns = capabilities, "—" = no paper covers it):
{gap_map}

Papers in the knowledge base (compact digest):
{digest}

Code that exists:
{repos}

Already proposed in earlier runs (don't repeat these unless you refine them into something clearly new):
{previous}

Propose 5-8 ideas. Reply with a JSON list of objects:
{{"title": "", "improvement": "what to change and why it should work, in plain words",
  "gap": "which gap-map cell or limitation it addresses", "evidence_papers": ["paper ids"],
  "closest_prior_work": ["paper ids or titles"], "feasibility": "sim vs real robot, rough effort",
  "rank": 1}}
rank 1 = most promising."""


@observe(name="ideas")
def run(topic: str, kb: KnowledgeBase, cfg: dict) -> list[dict]:
    gap_map = kb.root / "gap-map.md"
    previous = [{"title": i.get("title"), "novelty": i.get("novelty")} for i in kb.last_ideas()]
    prompt = PROMPT.format(
        topic=topic, lab=lab_context(cfg),
        gap_map=gap_map.read_text(encoding="utf-8") if gap_map.exists() else "(none yet)",
        digest=as_json(kb.digest()), repos=as_json(kb.repos_digest()),
        previous=as_json(previous) if previous else "(none)",
    )
    for attempt in range(1, ATTEMPTS + 1):
        reply = complete(prompt, system=SYSTEM, name="ideas", **model_args(cfg, "deep"))
        parsed = parse_json(reply.text, list)
        found = [i for i in parsed or [] if isinstance(i, dict) and i.get("title")]
        if found:
            # Novelty is the Critic's call, never the proposer's.
            result = rerank([{**normalize_idea(i), "novelty": "unclear"} for i in found])
            update_span(input={"topic": topic, "papers": len(kb.digest())}, metadata={"ideas": len(result)})
            return result
        logger.warning("Ideas: reply was not a JSON list of ideas (attempt %d)", attempt)
    logger.error("Ideas: no usable ideas after %d attempts", ATTEMPTS)
    return []


def normalize_idea(raw: dict) -> dict:
    lists = ("evidence_papers", "closest_prior_work")
    idea = {k: str(raw.get(k) or "") for k in ("title", "improvement", "gap", "novelty_evidence", "feasibility")}
    idea.update({k: [str(x) for x in raw.get(k) or [] if x] if isinstance(raw.get(k), list) else [] for k in lists})
    idea["novelty"] = raw.get("novelty") if raw.get("novelty") in NOVELTY else "unclear"
    try:
        idea["rank"] = int(raw.get("rank"))
    except (TypeError, ValueError):
        idea["rank"] = 99
    return idea


def rerank(ideas: list[dict]) -> list[dict]:
    """Sort by rank and renumber 1..n, so ranks are always clean."""
    ordered = sorted(ideas, key=lambda i: i["rank"])
    for n, idea in enumerate(ordered, 1):
        idea["rank"] = n
    return ordered
