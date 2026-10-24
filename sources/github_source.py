"""GitHub repository search (REST API)."""
import json
import logging
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

USER_AGENT = "ResearchAssistant/1.0"
API = "https://api.github.com/search/repositories"


def search_repos(query: str, *, per_page: int, token: str | None) -> list[dict]:
    """Most-starred repos matching `query`, as Repo dicts with no links or notes yet."""
    params = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": per_page})
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API}?{params}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.load(resp).get("items", [])
    return [
        {
            "full_name": item["full_name"],
            "url": item["html_url"],
            "description": item.get("description") or "",
            "stars": item.get("stargazers_count", 0),
            "updated": (item.get("pushed_at") or "")[:10],
            "language": item.get("language") or "",
            "linked_papers": [],
            "why_relevant": "",
        }
        for item in items
    ]


def pause(token: str | None) -> None:
    """Stay under the search API limit: 10 requests/min anonymous, 30/min with a token."""
    time.sleep(2.1 if token else 6.5)
