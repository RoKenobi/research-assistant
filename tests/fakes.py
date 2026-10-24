"""Test doubles shared across agent tests."""
import copy

from core.config import DEFAULTS
from core.llm import LLMResult


def make_cfg(tmp_path, **overrides) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(_vault=tmp_path, _root=tmp_path, vault_path=str(tmp_path))
    cfg.update(overrides)
    return cfg


class FakeLLM:
    """Stands in for core.llm.complete; replies are looked up by call name.

    A reply may be a string, a list of strings (consumed in order), or an
    exception instance to raise.
    """

    def __init__(self, replies: dict):
        self.replies = {k: (list(v) if isinstance(v, list) else v) for k, v in replies.items()}
        self.calls: list[dict] = []

    def __call__(self, prompt, *, model, system, effort="low", name="claude", timeout=600):
        self.calls.append({"prompt": prompt, "model": model, "system": system, "effort": effort, "name": name})
        reply = self.replies[name]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return LLMResult(text=reply, model=model, usage={"input": 1, "output": 1}, cost_usd=0.0)

    def named(self, name: str) -> list[dict]:
        return [c for c in self.calls if c["name"] == name]


def paper(pid: str, **kw) -> dict:
    p = {"id": pid, "title": f"Paper {pid}", "authors": ["A. Author"], "abstract": f"Abstract of {pid}.",
         "pdf_url": f"https://arxiv.org/pdf/{pid}", "published": "2026-01-01", "source": "arxiv",
         "content": None, "content_type": None}
    p.update(kw)
    return p


def repo(full_name: str, **kw) -> dict:
    r = {"full_name": full_name, "url": f"https://github.com/{full_name}", "description": "d", "stars": 10,
         "updated": "2026-09-01", "language": "Python", "linked_papers": [], "why_relevant": ""}
    r.update(kw)
    return r
