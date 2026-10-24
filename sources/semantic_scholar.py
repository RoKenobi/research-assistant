"""Semantic Scholar: which papers an arXiv paper cites and which cite it."""
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

USER_AGENT = "ResearchAssistant/1.0"
API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}?fields=references.externalIds,citations.externalIds"


def neighbors(arxiv_id: str, *, limit: int) -> list[str]:
    """arXiv ids of the paper's references and citations, deduped, at most `limit`.

    The free API is rate-limited and shared, so any failure just means no
    neighbours this time.
    """
    req = urllib.request.Request(API.format(id=arxiv_id), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as e:
        logger.warning("Semantic Scholar lookup failed for %s: %s", arxiv_id, e)
        return []

    ids, seen = [], {arxiv_id}
    for entry in (data.get("references") or []) + (data.get("citations") or []):
        found = ((entry or {}).get("externalIds") or {}).get("ArXiv")
        if found and found not in seen:
            seen.add(found)
            ids.append(found)
    return ids[:limit]
