"""arXiv Scout: topic -> new candidate papers, ranked and capped."""
import logging

from agents._shared import as_json, lab_context, model_args
from core.jsonparse import parse_json
from core.kb import KnowledgeBase
from core.llm import LLMUsageLimitError, complete
from core.tracing import observe, update_span
from sources import arxiv_source, semantic_scholar

logger = logging.getLogger(__name__)

MAX_QUERIES = 6
NEIGHBOR_SEEDS = 3
NEIGHBOR_LIMIT = 20

SYSTEM = "You are a research librarian for embodied AI and robotics. Reply with JSON only."

QUERY_PROMPT = """\
Write 4-6 arXiv search queries that together cover the research topic below: the core \
problem, the main method families, and adjacent capabilities a researcher improving \
this area would need to read about. Each query is a short keyword phrase (2-5 words).

Topic: {topic}
{lab}

Reply with a JSON list of strings."""

RANK_PROMPT = """\
Rank these candidate papers by how useful they are to a researcher who wants to improve \
methods in the topic below: directly on-topic method papers first, then strong baselines \
and benchmarks, then loosely related work. Drop clearly off-topic papers.

Topic: {topic}
{lab}

Candidates:
{candidates}

Reply with a JSON list of paper ids, best first."""


@observe(name="arxiv_scout")
def run(topic: str, kb: KnowledgeBase, cfg: dict, *, limit: int, queries: list[str] | None = None) -> list[dict]:
    followup = queries is not None
    if not followup:
        queries = _make_queries(topic, cfg)
    logger.info("arXiv Scout: queries %s", queries)

    pool = arxiv_source.search(queries, per_query=cfg["arxiv_results_per_query"],
                               category=cfg["arxiv_category_filter"])
    if followup:
        for p in pool:
            p["source"] = "followup"
    else:
        pool += _citation_neighbors(pool, kb)

    pool = [p for p in pool if not kb.has_paper(p["id"])]
    chosen = _rank(topic, pool, cfg)[:limit] if len(pool) > limit else pool
    update_span(input={"topic": topic, "queries": queries},
                metadata={"candidates": len(pool), "chosen": len(chosen), "followup": followup})
    logger.info("arXiv Scout: %d new candidates, keeping %d", len(pool), len(chosen))
    return chosen


def _make_queries(topic: str, cfg: dict) -> list[str]:
    try:
        reply = complete(QUERY_PROMPT.format(topic=topic, lab=lab_context(cfg)), system=SYSTEM,
                         name="arxiv_queries", **model_args(cfg, "fast"))
        parsed = parse_json(reply.text, list) or []
    except LLMUsageLimitError:
        raise
    except Exception as e:
        logger.warning("Query generation failed, searching the topic itself: %s", e)
        parsed = []
    queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()][:MAX_QUERIES]
    return queries or [topic]


def _citation_neighbors(pool: list[dict], kb: KnowledgeBase) -> list[dict]:
    """Papers that cite or are cited by the top hits, which keyword search misses."""
    known = {p["id"] for p in pool}
    new_ids: list[str] = []
    for p in pool[:NEIGHBOR_SEEDS]:
        for nid in semantic_scholar.neighbors(p["id"], limit=NEIGHBOR_LIMIT):
            if nid not in known and not kb.has_paper(nid):
                known.add(nid)
                new_ids.append(nid)
    return arxiv_source.fetch_by_ids(new_ids, source="semantic_scholar") if new_ids else []


def _rank(topic: str, pool: list[dict], cfg: dict) -> list[dict]:
    candidates = as_json([{"id": p["id"], "title": p["title"], "abstract": p["abstract"][:600]} for p in pool])
    reply = complete(RANK_PROMPT.format(topic=topic, lab=lab_context(cfg), candidates=candidates),
                     system=SYSTEM, name="arxiv_rank", **model_args(cfg, "fast"))
    by_id = {p["id"]: p for p in pool}
    ids = [i for i in parse_json(reply.text, list) or [] if isinstance(i, str)]
    ranked = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
    seen = {p["id"] for p in ranked}
    return ranked + [p for p in pool if p["id"] not in seen]
