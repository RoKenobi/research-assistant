import contextlib
import json
from types import SimpleNamespace

import pytest

from core import llm
from core.llm import LLMError, LLMUsageLimitError, check_cli, complete

OK_REPLY = {
    "result": "hi",
    "is_error": False,
    "total_cost_usd": 0.01,
    "usage": {"input_tokens": 3, "cache_creation_input_tokens": 600,
              "cache_read_input_tokens": 0, "output_tokens": 2},
    "modelUsage": {"claude-opus-5-5": {}},
}


class _Calls(list):
    def __init__(self):
        super().__init__()
        self.state = {"stdout": json.dumps(OK_REPLY), "returncode": 0}


@pytest.fixture
def fake_cli(monkeypatch):
    """Replace the `claude -p` subprocess; returns the list of recorded calls."""
    calls = _Calls()
    state = calls.state

    def fake_run(argv, **kw):
        calls.append({"argv": argv, **kw})
        return SimpleNamespace(stdout=state["stdout"], stderr="", returncode=state["returncode"])

    monkeypatch.setattr(llm.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    return calls


def test_complete_builds_cli_command(fake_cli):
    complete("the prompt", model="claude-opus-5-5", system="be terse", effort="high")
    argv = fake_cli[0]["argv"]
    assert argv[0] == "/bin/claude"
    assert "-p" in argv
    for flag, value in [("--output-format", "json"), ("--model", "claude-opus-5-5"),
                        ("--effort", "high"), ("--tools", ""), ("--system-prompt", "be terse")]:
        assert argv[argv.index(flag) + 1] == value
    for flag in ("--no-session-persistence", "--safe-mode", "--strict-mcp-config",
                 "--disable-slash-commands"):
        assert flag in argv
    assert "--bare" not in argv
    assert "the prompt" not in argv
    assert fake_cli[0]["input"] == "the prompt"


def test_complete_ignores_api_key_env(fake_cli, monkeypatch):
    # A stray ANTHROPIC_API_KEY (old .env, shell profile) would make claude -p bill
    # that key instead of the Claude plan, or fail with "Invalid API key".
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-placeholder")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("KEEP_ME", "1")
    complete("p", model="m", system="s")
    env = fake_cli[0]["env"]
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["KEEP_ME"] == "1"


def test_complete_parses_json_result(fake_cli):
    result = complete("p", model="claude-sonnet-5-5", system="s")
    assert result.text == "hi"
    assert result.model == "claude-opus-5-5"  # from modelUsage, i.e. what actually served it
    assert result.usage == {"input": 603, "output": 2}
    assert result.cost_usd == 0.01


def test_complete_raises_usage_limit(fake_cli):
    fake_cli.state["stdout"] = json.dumps(
        {"result": "Claude usage limit reached. Your limit will reset at 5pm", "is_error": True})
    with pytest.raises(LLMUsageLimitError):
        complete("p", model="m", system="s")


def test_complete_raises_llm_error_on_other_errors(fake_cli):
    fake_cli.state.update(stdout="", returncode=1)
    with pytest.raises(LLMError) as exc:
        complete("p", model="m", system="s")
    assert not isinstance(exc.value, LLMUsageLimitError)

    fake_cli.state.update(stdout=json.dumps(
        {"result": "Not logged in · Please run /login", "is_error": True}), returncode=1)
    with pytest.raises(LLMError) as exc:
        complete("p", model="m", system="s")
    assert not isinstance(exc.value, LLMUsageLimitError)
    assert "/login" in str(exc.value)


def test_check_cli_missing_exits(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        check_cli()
    assert "Claude Code" in str(exc.value)


def test_check_cli_not_logged_in_exits(fake_cli):
    fake_cli.state.update(stdout=json.dumps(
        {"result": "Not logged in · Please run /login", "is_error": True}), returncode=1)
    with pytest.raises(SystemExit) as exc:
        check_cli()
    assert "/login" in str(exc.value)


def test_complete_records_generation_when_tracing_enabled(monkeypatch, fake_cli):
    recorded = {}

    class FakeGen:
        def update(self, **kw):
            recorded.update(kw)

    @contextlib.contextmanager
    def fake_generation(name, **kw):
        recorded.update(name=name, start=kw)
        yield FakeGen()

    monkeypatch.setattr(llm, "generation", fake_generation)
    complete("hi", model="claude-sonnet-5-5", system="sys", name="reader")

    assert recorded["name"] == "reader"
    assert recorded["start"]["input"][0] == {"role": "system", "content": "sys"}
    assert recorded["model"] == "claude-opus-5-5"
    assert recorded["usage_details"] == {"input": 603, "output": 2}
    assert recorded["cost_details"] == {"total": 0.01}
    assert recorded["metadata"]["cost_basis"] == "api_equivalent_estimate"
