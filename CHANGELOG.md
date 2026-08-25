# Changelog

## 0.1.3 — 2026-08-25

Fourth adversarial audit (8 findings, all fixed). The four most impactful:

**Fixed (correctness)**
- **Session now inherits all five protections.** Previously the Session
  manager bypassed the truncation, retry, and starvation-aware caps that
  `ClientConfig` advertises — `Session._chat_with_history` had its own
  raw `urlopen` call that never used the retry loop or `slow_death_s`.
  Routed Session through the new shared `_do_request` path. The contract
  is now: Session and `chat()` go through the same request code, so any
  future change to one automatically applies to the other.
- **4xx errors now include the server's response body.** The previous
  `LocalModelUnavailable("HTTP 400: Bad Request")` was a debug
  nightmare when the server rejected a field — the body says WHAT
  ("unrecognized arguments: chat_template_kwargs""). When the body
  hints at `chat_template_kwargs` + thinking is on, the error now
  includes the actionable line: "try PreflightClient(thinking_off=False)".
- **Session.resume() now raises typed `SessionNotFoundError`** with the
  resolved path and a recovery hint, instead of a raw `FileNotFoundError`,
  for both unknown ids and corrupted checkpoint files.
- **Session.checkpoint_every=0** (documented as "omit checkpoints") no
  longer crashes with `ZeroDivisionError`. Treated as the no-op it
  should be.

**Polish**
- `__all__` indentation normalized; `pyproject` gains `project.urls`
  (Homepage/Changelog/Documentation/Issues) so the PyPI page links out.
- `memory.py`: page-size fallback now uses portable `os.sysconf` instead
  of an Apple-Silicon-specific constant (Intel-Mac correctness).
- README's `# llm-preflight` header and module banner both renamed;
  Session section gets a privacy note about default `~/.llm-preflight`
  checkpoint location.

**Tests:** 35 → 40 (5 new regression tests pinning all of the above).

## 0.1.2 — 2026-08-25

Published to PyPI as **local-llm-preflight** (the name `llm-preflight` is
owned by an unrelated project; import name stays `llm_preflight`). GitHub
repository renamed to match.

**Fixed**
- `client.py`: removed a nonexistent `PreflightConfig` from `__all__`
  (`from llm_preflight.client import *` raised AttributeError).
- README: issue quote now verbatim; hand-build note (`rm -rf build/`)
  added; clone instructions corrected for the renamed repo.
- Example TOML / docs: port guidance corrected — vLLM defaults to 8000,
  Ollama to 11434, llama.cpp to 8080. Set yours explicitly.

**Changed**
- Publishing is GitHub-OIDC only (zero tokens); the workflow now verifies
  the tag matches `pyproject.toml` before uploading, and skips files that
  already exist (idempotent re-releases).
- `MANIFEST.in`: tests included explicitly (was accidental via a legacy
  setuptools pattern that dropped `integration_live.py`).

## 0.1.1 — 2026-08-25

Pre-publication audit release (full-repo security/privacy/quality review).

**Fixed**
- `client.py`: 5xx HTTP responses now follow the starvation-aware retry
  policy. Previously *any* `HTTPError` — including `503` while a model
  loads, the most common real-world local-server hiccup — raised with zero
  retries, silently defeating the retry protection. 4xx still fails fast.
- `config.py`: an explicit config `path` that does not exist now raises
  `FileNotFoundError` instead of silently falling back to defaults — a
  typo'd config must not run unattended jobs against the wrong server.
- `probe.py`: stale usage string replaced with the installed-command form.

**Changed (tool-agnosticism)**
- Default `base_url` port changed to `8000` (vLLM's default; a common
  denominator — Ollama defaults to 11434, llama.cpp to 8080). The previous
  default encoded one author's personal port.
- Docs and examples use a neutral `your-served-model-id` placeholder;
  the real measured model names appear only in `docs/MEASURED.md`,
  explicitly labeled as the measurement rig.
- `docs/MEASURED.md`: business-context wording genericized.

**Removed**
- Dead files: root `server-capability-probe.py` (stale duplicate of the
  packaged probe) and `llm_preflight/probe_main.py` (unreferenced;
  `sys.exit()` at import time — hazardous to import-all tooling).

**Tests**
- 31 → 35: +3 regression tests for the 5xx retry policy; config tests
  updated for the explicit-path semantics; fixed a test-helper leak
  (`FakeTime` polluting later tests).

## 0.1.0 — 2026-08-25

Initial release.

- Cross-platform memory preflight: macOS unified memory (free + purgeable +
  inactive), Linux `MemAvailable`, Windows `GlobalMemoryStatusEx`, optional
  independent NVIDIA VRAM gate (pynvml / nvidia-smi).
- Starvation-aware retry: attempts dying after `slow_death_s` are never
  retried.
- Thinking-mode-off by default via `chat_template_kwargs` (measured 4.7×
  token / 4.1× wall-time reduction on reasoning-style models).
- Token budgets: hard input cap, bounded output, typed failures
  (`MemoryPressureError` vs `LocalModelUnavailable`).
- Experimental session manager: truncation-with-anchors compaction,
  turn-boundary checkpoints, crash resume via message-list replay.
- Server capability probe (`llm-preflight-probe`): checks usage reporting,
  thinking-kwarg support, real context caps on any OpenAI-compatible server.
- 31 platform-independent unit tests + live integration suite.
- `docs/MEASURED.md`: every default backed by real measurements.
