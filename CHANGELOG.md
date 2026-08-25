# Changelog

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
- Default `base_url` port standardized to `8000` (Ollama/llama.cpp/
  FreeToken/vLLM convention). The previous default encoded one author's
  personal port.
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
