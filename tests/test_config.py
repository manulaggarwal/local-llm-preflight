"""Config loading tests."""

import pytest

from llm_preflight.config import load_config

TOML_FULL = """
[server]
base_url = "http://127.0.0.1:8000/v1"
model = "Qwen3.6-35B-A3B"

[budgets]
max_input_chars = 60000
max_tokens = 2048
timeout_s = 300

[retry]
retries = 2
slow_death_s = 180.0

[memory]
min_system_mb = 3000
min_vram_mb = 2000
cold_system_mb = 8000

[behavior]
thinking_off = false
expect_json = true
"""


class TestLoadConfig:
    def test_missing_file_gives_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nope.toml")
        assert cfg.base_url.endswith("/v1")
        assert cfg.max_tokens == 1024

    def test_full_file_overrides_everything(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text(TOML_FULL)
        cfg = load_config(p)
        assert cfg.model == "Qwen3.6-35B-A3B"
        assert cfg.max_input_chars == 60000
        assert cfg.retries == 2
        assert cfg.slow_death_s == 180.0
        assert cfg.thinking_off is False
        assert cfg.expect_json is True
        assert cfg.min_vram_mb == 2000

    def test_partial_file_uses_defaults_for_rest(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text('[server]\nmodel = "x"\n')
        cfg = load_config(p)
        assert cfg.model == "x"
        assert cfg.max_tokens == 1024  # default intact

    def test_null_cold_disables_gate(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text('[memory]\ncold_system_mb = 0\n')  # TOML null is untyped; 0 means 0
        cfg = load_config(p)
        assert cfg.cold_system_mb == 0
