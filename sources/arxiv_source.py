"""arXiv: search over queries, look up ids, and fetch a paper's body.

The body is the LaTeX source when arXiv has it (equations and all), else the
PDF's text, else just the abstract.
"""
import gzip
import io
import logging
import re
import tarfile
import urllib.request

import arxiv
import pymupdf

logger = logging.getLogger(__name__)

USER_AGENT = "ResearchAssistant/1.0"
MAX_TERMS = 6  # AND-ing more terms than this almost never matches anything
_STOPWORDS = {"a", "an", "and", "or", "the", "of", "for", "in", "on", "with", "to", "via", "by", "using", "from"}


def _client() -> arxiv.Client:
    return arxiv.Client(delay_seconds=3.0)


def _to_paper(result, source: str) -> dict:
    return {
        # Version-free id ("2307.15818v2" -> "2307.15818") so versions dedupe as one paper.
        "id": re.sub(r"v\d+$", "", result.get_short_id()),
        "title": result.title,
        "authors": [a.name for a in result.authors[:3]],
        "abstract": result.summary.replace("\n", " "),
        "pdf_url": result.pdf_url,
        "published": str(result.published.date()) if result.published else "",
        "source": source,
        "content": None,
        "content_type": None,
    }


def search(queries: list[str], *, per_query: int, category: str) -> list[dict]:
    """Relevance-ranked arXiv hits for each query, deduped by id, in query order."""
    client = _client()
    scope = f" AND {category}" if category else ""
    papers, seen = [], set()
    for query in queries:
        terms = _terms(query)
        if not terms:
            continue
        search_ = arxiv.Search(query=f"{terms}{scope}", max_results=per_query,
                               sort_by=arxiv.SortCriterion.Relevance)
        try:
            for result in client.results(search_):
                paper = _to_paper(result, "arxiv")
                if paper["id"] not in seen:
                    seen.add(paper["id"])
                    papers.append(paper)
        except Exception as e:
            logger.warning("arXiv search failed for %r: %s", query, e)
    return papers


def _terms(query: str) -> str:
    """'whole-body control for humanoids' -> 'all:whole-body AND all:control AND all:humanoids'.

    A quoted multi-word phrase only matches that exact wording, so a 5-word
    query almost always returns nothing; AND-ed terms keep it specific without that.
    """
    words = [w for w in re.findall(r"[\w-]+", query.lower()) if w not in _STOPWORDS and w.strip("-")]
    return " AND ".join(f"all:{w}" for w in words[:MAX_TERMS])


def fetch_by_ids(ids: list[str], source: str = "arxiv") -> list[dict]:
    """Paper records for known arXiv ids (inbox links, citation neighbours)."""
    if not ids:
        return []
    try:
        return [_to_paper(r, source) for r in _client().results(arxiv.Search(id_list=ids))]
    except Exception as e:
        logger.warning("arXiv id lookup failed for %s: %s", ids, e)
        return []


def fetch_content(paper: dict, char_budget: int) -> None:
    """Fill paper["content"] / ["content_type"]: LaTeX, then PDF text, then the abstract."""
    if not paper["id"].startswith("local-"):
        latex = _fetch_latex_source(paper["id"], char_budget)
        if latex:
            paper["content"], paper["content_type"] = latex, "latex"
            return
        text = _fetch_pdf_text(paper.get("pdf_url") or "", char_budget)
        if text:
            paper["content"], paper["content_type"] = text, "pdf_text"
            return
    paper["content"], paper["content_type"] = paper.get("abstract") or "", "abstract"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _fetch_latex_source(arxiv_id: str, char_budget: int) -> str:
    try:
        return extract_main_tex(_download(f"https://arxiv.org/src/{arxiv_id}"), char_budget)
    except Exception as e:
        logger.warning("LaTeX fetch failed for %s: %s", arxiv_id, e)
    return ""


def extract_main_tex(raw: bytes, char_budget: int) -> str:
    """Return the paper body from an arXiv source download.

    arXiv serves a gzipped tarball for multi-file papers, a gzipped single .tex
    for one-file papers, and sometimes just a PDF.
    """
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw

    files: dict[str, str] = {}
    buf = io.BytesIO(data)
    if tarfile.is_tarfile(buf):
        buf.seek(0)
        with tarfile.open(fileobj=buf) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        name = member.name.removeprefix("./")
                        files[name] = strip_comments(f.read().decode("utf-8", errors="ignore"))
    elif not data.startswith(b"%PDF"):
        files["main.tex"] = strip_comments(data.decode("utf-8", errors="ignore"))

    if not files:
        return ""
    # Prefer the file that actually declares the document; else the largest.
    mains = [n for n, t in files.items() if "\\documentclass" in t] or list(files)
    main = max(mains, key=lambda n: len(files[n]))
    body = clean_latex(inline_inputs(files[main], files))
    if len(body) < 500:
        return ""
    if len(body) > char_budget:
        logger.info("LaTeX body is %d chars, truncating to %d", len(body), char_budget)
    return body[:char_budget]


_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


def inline_inputs(tex: str, files: dict[str, str], depth: int = 0) -> str:
    r"""Replace \input{x} / \include{x} with the referenced file from the tarball.

    Many papers keep every section in its own file, so the main .tex alone is
    just front matter.
    """
    if depth > 5:
        return tex

    def resolve(match: re.Match) -> str:
        ref = match.group(1).strip().removeprefix("./")
        for candidate in (ref, f"{ref}.tex"):
            if candidate in files:
                return inline_inputs(files[candidate], files, depth + 1)
        return match.group(0)

    return _INPUT_RE.sub(resolve, tex)


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)  # keep escaped \%


def clean_latex(tex: str) -> str:
    """Drop the preamble, bibliography and comments so the budget is spent on content."""
    tex = strip_comments(tex)
    begin = tex.find("\\begin{document}")
    if begin != -1:
        tex = tex[begin + len("\\begin{document}"):]
    for marker in ("\\begin{thebibliography}", "\\bibliography{", "\\end{document}"):
        end = tex.find(marker)
        if end != -1:
            tex = tex[:end]
    return re.sub(r"\n\s*\n+", "\n\n", tex).strip()


def _fetch_pdf_text(pdf_url: str, char_budget: int) -> str:
    if not pdf_url:
        return ""
    try:
        return pdf_text_from_bytes(_download(pdf_url), char_budget)
    except Exception as e:
        logger.warning("PDF download failed for %s: %s", pdf_url, e)
    return ""


def pdf_text_from_bytes(data: bytes, char_budget: int) -> str:
    """Plain text from a PDF, stopping once the budget is reached."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages, total = [], 0
            for page in doc:
                text = page.get_text()
                pages.append(text)
                total += len(text)
                if total > char_budget:
                    break
        return "\n".join(pages)[:char_budget]
    except Exception as e:
        logger.warning("PDF text extraction failed: %s", e)
    return ""
