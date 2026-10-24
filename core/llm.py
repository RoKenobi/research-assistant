import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from core.tracing import generation

logger = logging.getLogger(__name__)

# `claude -p` runs on the user's Claude plan, so there is no API key to manage.
# --safe-mode skips the user's plugins, hooks, skills and CLAUDE.md, which
# otherwise add ~68K input tokens to every call. Never use --bare: it skips the
# keychain read and fails with "Not logged in".
_FIXED_FLAGS = [
    "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
    "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
]
# If either is set, claude -p bills that key instead of the user's plan login.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_USAGE_LIMIT_RE = re.compile(r"usage limit|rate limit|limit (?:reached|will reset)", re.I)


class LLMError(RuntimeError):
    pass


class LLMUsageLimitError(LLMError):
    """The Claude plan's usage limit was hit; the run should stop and resume later."""


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict[str, int]
    cost_usd: float | None


def _executable() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise SystemExit(
            "Claude Code (the `claude` command) was not found. Install it from "
            "https://claude.com/claude-code, then run `claude` and /login."
        )
    return exe


def check_cli() -> str:
    """Fail fast, before any fetching, if Claude Code is missing or logged out."""
    exe = _executable()
    try:
        complete("Reply with exactly: ok", model="claude-sonnet-5-5", system="Reply tersely.",
                 effort="low", name="check_cli", timeout=120)
    except LLMError as e:
        raise SystemExit(f"Claude Code is installed but the test call failed: {e}\n"
                         "Run `claude` and /login, then try again.") from e
    return exe


def complete(
    prompt: str,
    *,
    model: str,
    system: str,
    effort: str = "low",
    name: str = "claude",
    timeout: int = 600,
) -> LLMResult:
    """Send one prompt through `claude -p` and return its text and usage.

    The prompt goes on stdin: Windows caps command lines at ~32K characters and
    paper bodies are longer.
    """
    argv = [_executable(), *_FIXED_FLAGS, "--model", model, "--effort", effort,
            "--system-prompt", system]
    trace_input = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    with generation(name, model=model, input=trace_input, model_parameters={"effort": effort}) as gen:
        try:
            result = _run(argv, prompt, timeout)
        except LLMError as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        gen.update(
            model=result.model,
            output=result.text,
            usage_details=result.usage,
            cost_details={"total": result.cost_usd} if result.cost_usd is not None else None,
            # Billed to the Claude plan; the CLI's dollar figure is what the API would charge.
            metadata={"cost_basis": "api_equivalent_estimate"},
        )
    return result


def _run(argv: list[str], prompt: str, timeout: int) -> LLMResult:
    try:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        proc = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                              encoding="utf-8", timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"claude -p timed out after {timeout}s") from e

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "").strip()[:500]
        raise LLMError(f"claude -p exited {proc.returncode} without JSON output: {detail}") from None

    text = data.get("result") or ""
    if data.get("is_error") or proc.returncode != 0:
        if _USAGE_LIMIT_RE.search(text):
            raise LLMUsageLimitError(text)
        raise LLMError(text or f"claude -p exited {proc.returncode}")

    usage = data.get("usage") or {}
    served_by = next(iter(data.get("modelUsage") or {}), None)
    return LLMResult(
        text=text,
        model=served_by or argv[argv.index("--model") + 1],
        usage={
            "input": sum(usage.get(k, 0) for k in
                         ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "output": usage.get("output_tokens", 0),
        },
        cost_usd=data.get("total_cost_usd"),
    )
