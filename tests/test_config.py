import pytest

from core import config


def _write_config(tmp_path, text):
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")


def test_load_config_fills_defaults(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_config(tmp_path, f'vault_path: "{vault.as_posix()}"\n')
    monkeypatch.setattr(config, "ROOT", tmp_path)

    cfg = config.load_config()
    assert cfg["_vault"] == vault
    assert cfg["research_folder"] == "Research"
    assert cfg["lab"] == {"robots": ["Unitree G1", "AgiBot X2"], "simulators": ["MuJoCo"]}
    assert cfg["models"] == {"fast": "claude-sonnet-5-5", "deep": "claude-opus-5-5"}
    assert cfg["effort"] == {"fast": "low", "deep": "high"}
    assert cfg["arxiv_category_filter"] == "(cat:cs.RO OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV)"
    assert (cfg["arxiv_results_per_query"], cfg["max_new_papers"], cfg["followup_papers"]) == (8, 15, 5)
    assert (cfg["max_critic_rounds"], cfg["max_repos"], cfg["paper_char_budget"]) == (2, 10, 40000)
    assert (cfg["llm_concurrency"], cfg["llm_timeout_sec"]) == (2, 600)


def test_load_config_merges_nested_overrides(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_config(tmp_path, f'vault_path: "{vault.as_posix()}"\nmodels:\n  deep: claude-fable-5-1\nmax_new_papers: 3\n')
    monkeypatch.setattr(config, "ROOT", tmp_path)

    cfg = config.load_config()
    assert cfg["models"] == {"fast": "claude-sonnet-5-5", "deep": "claude-fable-5-1"}
    assert cfg["max_new_papers"] == 3


def test_load_config_keeps_vault_guard(tmp_path, monkeypatch):
    _write_config(tmp_path, f'vault_path: "{(tmp_path / "missing").as_posix()}"\n')
    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        config.load_config()
