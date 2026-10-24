import copy
import yaml
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent

DEFAULTS = {
    "research_folder": "Research",
    # Used to rank relevance and judge feasibility; other simulators still count.
    "lab": {"robots": ["Unitree G1", "AgiBot X2"], "simulators": ["MuJoCo"]},
    "models": {"fast": "claude-sonnet-5-5", "deep": "claude-opus-5-5"},
    "effort": {"fast": "low", "deep": "high"},
    "arxiv_category_filter": "(cat:cs.RO OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV)",
    "arxiv_results_per_query": 8,
    "max_new_papers": 15,
    "followup_papers": 5,
    "max_critic_rounds": 2,
    "max_repos": 10,
    "paper_char_budget": 40000,
    "llm_concurrency": 2,
    "llm_timeout_sec": 600,
}


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in loaded.items():
        # One level deep, so overriding models.deep keeps models.fast.
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["_root"] = ROOT
    cfg["_vault"] = Path(cfg["vault_path"]).expanduser()
    # Fail before spending plan usage, and never mkdir a phantom vault on a typo
    # (or before OneDrive has mounted).
    if not cfg["_vault"].is_dir():
        raise SystemExit(f"vault_path does not exist: {cfg['_vault']} — fix it in config.yaml")
    return cfg
