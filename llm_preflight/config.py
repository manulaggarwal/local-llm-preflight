"""Load ClientConfig from a TOML file so users configure per-hardware.

Example ``llm-preflight.toml`` (every key optional — defaults live in
ClientConfig):

    [server]
    base_url = "http://127.0.0.1:8000/v1"
    model = "your-served-model-id"

    [budgets]
    max_input_chars = 60000
    max_tokens = 2048
    temperature = 0.4
    timeout_s = 600

    [retry]
    retries = 1
    slow_death_s = 90.0          # raise for interactive long generations

    [memory]
    min_system_mb = 2500
    min_vram_mb = 1500
    cold_system_mb = 7000        # omit to keep default; disable only in code:
                                # ClientConfig(cold_system_mb=None) — TOML has no null

    [behavior]
    thinking_off = true
    expect_json = false
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .client import ClientConfig

_DEFAULT_PATHS = (
    Path("llm-preflight.toml"),
    Path.home() / ".config" / "llm-preflight.toml",
)


def load_config(path: str | Path | None = None) -> ClientConfig:
    """Build a ClientConfig from TOML.

    ``path=None`` searches default locations, falling back to defaults when
    none exist. An explicit ``path`` that does not exist raises
    FileNotFoundError — a typo'd config must not silently fail open against
    the wrong server/model in unattended jobs.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        candidates = [p]
    else:
        candidates = [p for p in _DEFAULT_PATHS if p.exists()]
    if not candidates:
        return ClientConfig()

    with open(candidates[0], "rb") as f:
        data = tomllib.load(f)

    server = data.get("server", {})
    budgets = data.get("budgets", {})
    retry = data.get("retry", {})
    memory = data.get("memory", {})
    behavior = data.get("behavior", {})

    return ClientConfig(
        base_url=server.get("base_url", "http://127.0.0.1:8000/v1"),
        model=server.get("model", ""),
        timeout_s=budgets.get("timeout_s", 600),
        max_input_chars=budgets.get("max_input_chars", 96_000),
        max_tokens=budgets.get("max_tokens", 1024),
        temperature=budgets.get("temperature", 0.4),
        retries=retry.get("retries", 1),
        slow_death_s=retry.get("slow_death_s", 90.0),
        thinking_off=behavior.get("thinking_off", True),
        expect_json=behavior.get("expect_json", False),
        min_system_mb=memory.get("min_system_mb", 2500),
        min_vram_mb=memory.get("min_vram_mb", 1500),
        cold_system_mb=memory.get("cold_system_mb", 7000),
    )
