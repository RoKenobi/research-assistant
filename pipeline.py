import contextvars
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from agents import critic, ideas, reader, scout_arxiv, scout_github
from agents.reader import ReaderError
from core.config import load_config, load_env
from core.kb import KnowledgeBase
from core.llm import LLMUsageLimitError, check_cli
from core.tracing import flush, init_langfuse, observe, trace_attributes, update_span

# Windows consoles and redirected output default to the ANSI code page; the
# box-drawing characters below would raise UnicodeEncodeError there.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

USAGE = 'Usage: python pipeline.py "<topic>"   e.g. python pipeline.py "humanoid whole-body control"'


@dataclass
class RunResult:
    ideas_path: Path | None
    papers_read: int
    repos_found: int
    critic_rounds: int
    stopped_early: str | None


@observe(name="research_run")
def run(topic: str, cfg: dict) -> RunResult:
    with trace_attributes(trace_name=f"research | {topic}", metadata={"topic": topic}):
        kb = KnowledgeBase(cfg, topic)
        try:
            return _run(topic, cfg, kb)
        except LLMUsageLimitError:
            kb.write_gap_map()  # keep the map in step with whatever was read before stopping
            raise


def _run(topic: str, cfg: dict, kb: KnowledgeBase) -> RunResult:
    logger.info("━━━ Inbox + scouts ━━━")
    papers = kb.ingest_inbox()
    if papers:
        logger.info("Inbox: %d paper(s) of your own", len(papers))
    papers += scout_arxiv.run(topic, kb, cfg, limit=cfg["max_new_papers"] - cfg["followup_papers"])
    repos = scout_github.run(topic, kb, cfg, papers=papers)

    logger.info("━━━ Reader: %d paper(s) ━━━", len(papers))
    papers_read = _read_all(papers, topic, kb, cfg)
    repos_found = _add_repos(repos, kb)
    kb.write_gap_map()

    logger.info("━━━ Ideas ━━━")
    current = ideas.run(topic, kb, cfg)

    rounds = 0
    for rounds in range(1, cfg["max_critic_rounds"] + 1):
        final = rounds == cfg["max_critic_rounds"]
        logger.info("━━━ Critic round %d%s ━━━", rounds, " (final)" if final else "")
        current, queries = critic.run(topic, current, kb, cfg, final=final)
        if not queries:
            break
        logger.info("Critic asked for targeted searches: %s", queries)
        followups = scout_arxiv.run(topic, kb, cfg, limit=cfg["followup_papers"], queries=queries)
        followup_repos = scout_github.run(topic, kb, cfg, papers=followups, queries=queries)
        papers_read += _read_all(followups, topic, kb, cfg)
        repos_found += _add_repos(followup_repos, kb)
        kb.write_gap_map()

    meta = {"papers_read": papers_read, "repos_found": repos_found, "critic_rounds": rounds, "stopped_early": None}
    path = kb.write_ideas(current, meta)
    update_span(output={"ideas_path": str(path), **meta})
    return RunResult(path, papers_read, repos_found, rounds, None)


def _read_all(papers: list[dict], topic: str, kb: KnowledgeBase, cfg: dict) -> int:
    """Read papers concurrently; save each one as soon as it's done, so a stop loses nothing."""
    if not papers:
        return 0
    read = 0
    with ThreadPoolExecutor(max_workers=cfg["llm_concurrency"]) as pool:
        # copy_context() carries the Langfuse span into the worker thread, so
        # reader traces nest under this run.
        futures = {pool.submit(contextvars.copy_context().run, reader.run, p, topic, kb, cfg): p for p in papers}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                breakdown = future.result()
            except LLMUsageLimitError:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            except ReaderError as e:
                logger.warning("Skipping %s: %s", paper["id"], e)
                continue
            except Exception:
                logger.exception("Skipping %s: reader failed", paper["id"])
                continue
            kb.add_paper(paper, breakdown)
            if paper["source"] == "inbox":
                kb.mark_inbox_done(paper)
            read += 1
            logger.info("Read %s: %s", paper["id"], paper["title"][:80])
    return read


def _add_repos(repos: list[dict], kb: KnowledgeBase) -> int:
    for r in repos:
        kb.add_repo(r)
    return len(repos)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(USAGE)
        return 1
    topic = " ".join(argv).strip()

    load_env()
    cfg = load_config()
    check_cli()
    init_langfuse()

    logger.info("Research run: %s (models: %s / %s)", topic, cfg["models"]["fast"], cfg["models"]["deep"])
    try:
        result = run(topic, cfg)
    except LLMUsageLimitError:
        logger.error("Stopped: Claude plan usage limit reached. Progress is saved; "
                     "run the same command again after the limit resets.")
        return 2
    except Exception:
        logger.exception("Run failed")
        return 1
    finally:
        flush()

    print("\n─── Done ───")
    print(f"  Ideas: {result.ideas_path}")
    print(f"  Papers read: {result.papers_read} · repos: {result.repos_found} · critic rounds: {result.critic_rounds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
