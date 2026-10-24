import gzip
import io
import tarfile

from agents.arxiv_agent import clean_latex, extract_main_tex

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
