import gzip
import io
import json
import tarfile
import urllib.error
from types import SimpleNamespace

import arxiv
import pytest

from sources import arxiv_source, github_source, semantic_scholar
from sources.arxiv_source import clean_latex, extract_main_tex

BODY = "We propose a method. " * 40 + r"$$ L = \sum_i \ell(x_i) $$ costs 5\% less."
PAPER = (
    "\\documentclass{article}\n\\usepackage{amsmath}\n% secret comment\n"
    f"\\begin{{document}}\n{BODY}\n\\bibliography{{refs}}\n\\end{{document}}\n"
)


# --- LaTeX extraction -------------------------------------------------------

def _tar_gz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_clean_latex_strips_preamble_comments_and_bibliography():
    out = clean_latex(PAPER)
    assert "usepackage" not in out and "secret comment" not in out
    assert "bibliography" not in out
    assert r"\sum_i" in out and r"5\% less" in out


def test_extract_main_tex_prefers_documentclass_file_over_larger_include():
    big_include = "x" * 50000
    raw = _tar_gz({"appendix.tex": big_include, "main.tex": PAPER})
    assert extract_main_tex(raw, 20000).startswith("We propose")


def test_extract_main_tex_accepts_gzipped_single_file():
    assert r"\sum_i" in extract_main_tex(gzip.compress(PAPER.encode()), 20000)


def test_extract_main_tex_rejects_pdf_and_respects_budget():
    assert extract_main_tex(gzip.compress(b"%PDF-1.5 ..."), 20000) == ""
    assert len(extract_main_tex(gzip.compress(PAPER.encode()), 600)) == 600


def test_extract_main_tex_inlines_section_files():
    main = (
        "\\documentclass{article}\n\\begin{document}\n\\input{sections/intro}\n"
        "% \\input{sections/unused}\n\\end{document}\n"
    )
    raw = _tar_gz({"./main.tex": main, "./sections/intro.tex": BODY, "./sections/unused.tex": "NOPE" * 200})
    out = extract_main_tex(raw, 20000)
    assert out.startswith("We propose") and "NOPE" not in out


# --- arXiv search and content ------------------------------------------------

def _result(arxiv_id: str):
    return SimpleNamespace(
        get_short_id=lambda: arxiv_id, title=f"Paper {arxiv_id}",
        authors=[SimpleNamespace(name="A. Author")], summary="An abstract.\nSecond line.",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}", published=None,
    )


def test_search_builds_scoped_query(monkeypatch):
    queries = []

    def fake_results(self, search):
        queries.append(search.query)
        return iter([_result("2401.00001")])

    monkeypatch.setattr(arxiv.Client, "results", fake_results)
    papers = arxiv_source.search(["humanoid stairs", "push recovery"], per_query=8,
                                 category="(cat:cs.RO OR cat:cs.LG)")
    assert queries[0] == 'all:"humanoid stairs" AND (cat:cs.RO OR cat:cs.LG)'
    assert [p["id"] for p in papers] == ["2401.00001"]  # same id from both queries -> one paper
    paper = papers[0]
    assert paper["source"] == "arxiv" and paper["content"] is None and paper["content_type"] is None
    assert paper["abstract"] == "An abstract. Second line."


def test_fetch_content_falls_back_to_abstract(monkeypatch):
    monkeypatch.setattr(arxiv_source, "_fetch_latex_source", lambda arxiv_id, budget: "")
    monkeypatch.setattr(arxiv_source, "_fetch_pdf_text", lambda url, budget: "")
    paper = {"id": "2401.00001", "abstract": "Just the abstract.", "pdf_url": "u",
             "content": None, "content_type": None}
    arxiv_source.fetch_content(paper, 40000)
    assert paper["content"] == "Just the abstract."
    assert paper["content_type"] == "abstract"


def test_fetch_content_prefers_latex(monkeypatch):
    monkeypatch.setattr(arxiv_source, "_fetch_latex_source", lambda arxiv_id, budget: "tex body")
    paper = {"id": "2401.00001", "abstract": "a", "pdf_url": "u", "content": None, "content_type": None}
    arxiv_source.fetch_content(paper, 40000)
    assert (paper["content"], paper["content_type"]) == ("tex body", "latex")


# --- GitHub -------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_github_search_parses_items(monkeypatch):
    seen = []
    body = {"items": [{"full_name": "a/b", "html_url": "https://github.com/a/b", "description": None,
                       "stargazers_count": 5, "pushed_at": "2026-09-01T00:00:00Z", "language": "Python"}]}

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(github_source.urllib.request, "urlopen", fake_urlopen)
    repos = github_source.search_repos("humanoid wbc", per_page=5, token="t")
    assert repos == [{"full_name": "a/b", "url": "https://github.com/a/b", "description": "",
                      "stars": 5, "updated": "2026-09-01", "language": "Python",
                      "linked_papers": [], "why_relevant": ""}]
    req = seen[0]
    assert req.get_header("Authorization") == "Bearer t"
    assert "q=humanoid+wbc" in req.full_url and "sort=stars" in req.full_url and "per_page=5" in req.full_url


def test_github_search_without_token_has_no_auth_header(monkeypatch):
    seen = []
    monkeypatch.setattr(github_source.urllib.request, "urlopen",
                        lambda req, timeout=None: seen.append(req) or _FakeResponse(b'{"items": []}'))
    assert github_source.search_repos("x", per_page=5, token=None) == []
    assert seen[0].get_header("Authorization") is None


def test_github_pause_depends_on_token(monkeypatch):
    slept = []
    monkeypatch.setattr(github_source.time, "sleep", slept.append)
    github_source.pause(None)
    github_source.pause("t")
    assert slept == [6.5, 2.1]


# --- Semantic Scholar -----------------------------------------------------------

def test_semantic_scholar_neighbors_extracts_arxiv_ids(monkeypatch):
    body = {"references": [{"externalIds": {"ArXiv": "2101.00001"}}, {"externalIds": {}},
                           {"externalIds": None}],
            "citations": [{"externalIds": {"ArXiv": "2201.00002"}}, {"externalIds": {"ArXiv": "2307.15818"}},
                          {"externalIds": {"ArXiv": "2101.00001"}}]}
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        return _FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(semantic_scholar.urllib.request, "urlopen", fake_urlopen)
    assert semantic_scholar.neighbors("2307.15818", limit=20) == ["2101.00001", "2201.00002"]
    assert "paper/arXiv:2307.15818" in seen[0]
    assert semantic_scholar.neighbors("2307.15818", limit=1) == ["2101.00001"]


def test_semantic_scholar_http_error_returns_empty(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(semantic_scholar.urllib.request, "urlopen", boom)
    assert semantic_scholar.neighbors("2307.15818", limit=20) == []
