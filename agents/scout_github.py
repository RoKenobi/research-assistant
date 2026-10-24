"""GitHub Scout: topic and papers -> relevant repos, linked to the papers they implement."""
import logging
import os
import re

from agents._shared import as_json, lab_context, model_args
from core.jsonparse import parse_json
from core.kb import KnowledgeBase
from core.llm import LLMUsageLimitError, complete
from core.tracing import observe, update_span
from sources import github_source

logger = logging.getLogger(__name__)

MAX_QUERIES = 4
_ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}")

SYSTEM = "You are a research engineer who knows the embodied AI and robotics open-source ecosystem. Reply with JSON only."

QUERY_PROMPT = """\
Write 3-4 GitHub repository search queries for code a researcher improving the topic below \
would want: implementations, controllers or policies, simulation environments, benchmarks \
and datasets. Each query is 2-4 keywords.

Topic: {topic}
{lab}

Reply with a JSON list of strings."""

FILTER_PROMPT = """\
From these GitHub repositories, keep only the ones that are useful to a researcher improving \
methods in the topic below. For each one you keep, say in one sentence why it matters \
(what it implements, which robot or simulator, whether it is a baseline to build on).

Topic: {topic}
{lab}

Repositories:
{repos}

Reply with a JSON list of objects: {{"full_name": "...", "why_relevant": "..."}}, most useful first."""


@observe(name="github_scout")
def run(topic: str, kb: KnowledgeBase, cfg: dict, *, papers: list[dict],
        queries: list[str] | None = None) -> list[dict]:
    token = os.environ.get("GITHUB_TOKEN") or None
    queries = queries if queries is not None else _make_queries(topic, cfg)
    searches = [(q, 5, None) for q in queries]
    # A README that mentions a paper's arXiv id is usually its official code.
    searches += [(f'"{p["id"]}" in:readme', 3, p["id"]) for p in papers if _ARXIV_ID_RE.fullmatch(p["id"])]

    found: dict[str, dict] = {}
    for i, (query, per_page, paper_id) in enumerate(searches):
        if i:
            github_source.pause(token)
        try:
            hits = github_source.search_repos(query, per_page=per_page, token=token)
        except Exception as e:
            logger.warning("GitHub search failed for %r: %s", query, e)
            continue
        for hit in hits:
            entry = found.setdefault(hit["full_name"], hit)
            if paper_id and paper_id not in entry["linked_papers"]:
                entry["linked_papers"].append(paper_id)

    candidates = [r for r in found.values() if not kb.has_repo(r["full_name"])]
    kept = _filter(topic, candidates, cfg)[: cfg["max_repos"]] if candidates else []
    update_span(input={"topic": topic, "queries": queries},
                metadata={"candidates": len(candidates), "kept": len(kept)})
    logger.info("GitHub Scout: %d new repos, keeping %d", len(candidates), len(kept))
    return kept


def _make_queries(topic: str, cfg: dict) -> list[str]:
    try:
        reply = complete(QUERY_PROMPT.format(topic=topic, lab=lab_context(cfg)), system=SYSTEM,
                         name="github_queries", **model_args(cfg, "fast"))
        parsed = parse_json(reply.text, list) or []
    except LLMUsageLimitError:
        raise
    except Exception as e:
        logger.warning("GitHub query generation failed, searching the topic itself: %s", e)
        parsed = []
    queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()][:MAX_QUERIES]
    return queries or [topic]


def _filter(topic: str, candidates: list[dict], cfg: dict) -> list[dict]:
    listing = as_json([{k: r[k] for k in ("full_name", "description", "stars", "language", "linked_papers")}
                       for r in candidates])
    reply = complete(FILTER_PROMPT.format(topic=topic, lab=lab_context(cfg), repos=listing),
                     system=SYSTEM, name="github_filter", **model_args(cfg, "fast"))
    by_name = {r["full_name"]: r for r in candidates}
    kept = []
    for item in parse_json(reply.text, list) or []:
        name = item.get("full_name") if isinstance(item, dict) else None
        if name in by_name and by_name[name] not in kept:
            by_name[name]["why_relevant"] = str(item.get("why_relevant") or "")
            kept.append(by_name[name])
    return kept
