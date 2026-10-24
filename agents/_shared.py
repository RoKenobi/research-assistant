"""Helpers every agent uses: the lab context line and model selection."""
import json


def lab_context(cfg: dict) -> str:
    lab = cfg["lab"]
    return (
        f"The user's lab works with {' and '.join(lab['robots'])} robots, simulating in "
        f"{' / '.join(lab['simulators'])} by default. The simulator is not fixed: work in other "
        "simulators (Isaac Lab, Gazebo, ...) is still relevant, only rank directly portable work a "
        "little higher."
    )


def model_args(cfg: dict, tier: str) -> dict:
    """Keyword arguments for core.llm.complete for the 'fast' or 'deep' tier."""
    return {"model": cfg["models"][tier], "effort": cfg["effort"][tier], "timeout": cfg["llm_timeout_sec"]}


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)
