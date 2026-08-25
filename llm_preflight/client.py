"""The disciplined client for local LLM servers.

What this client enforces, and why (all measured on real hardware — see
docs/MEASURED.md):

1. Preflight memory check before every call. A starved local server doesn't
   error cleanly — it thrashes for minutes, times out, and a naive retry
   burns the same wall-clock again. Preflight converts that into a cheap,
   immediate, typed failure.
2. Starvation-aware retry: if an attempt died after ``slow_death_s`` seconds,
   it is NEVER retried — the failure was resource starvation, not a transient
   glitch, and retrying doubles the cost of a doomed call.
3. Thinking-mode-off by default. Reasoning-style models emit hidden
   <think> blocks that eat 4-5x tokens and wall-time on tasks that don't
   need them. Applied via ``chat_template_kwargs`` — prompt-level begging
   ("/no_think") measurably does NOT work.
4. Input budget: a hard character cap prevents accidental 60k-token prompts
   into a server that will take minutes to prefill.
5. Fresh context per call: statelessness is a feature for unattended work.

Works with any OpenAI-compatible server (omlx, llama.cpp, LM Studio,
Ollama, FreeToken, vLLM), with capability differences surfaced — not
silently assumed away. Use :mod:`llm_preflight.probe` to check a server.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .memory import check as memory_check

__all__ = [
    "PreflightClient",
    "MemoryPressureError",
    "LocalModelUnavailable",
    "ClientConfig",
]


class MemoryPressureError(RuntimeError):
    """System memory too low for local inference right now.

    Callers must NOT retry immediately — defer, schedule, or fall back.
    """


class LocalModelUnavailable(RuntimeError):
    """Server unreachable or errored after the allowed retry budget."""


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_BLOCK.sub("", text).strip()


class ClientConfig:
    """Connection + budget settings. Override anything; defaults are sane."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "",
        timeout_s: int = 600,
        max_input_chars: int = 96_000,        # ~24k tokens
        max_tokens: int = 1024,
        temperature: float = 0.4,
        retries: int = 1,
        slow_death_s: float = 90.0,           # > this = starvation, never retry
        thinking_off: bool = True,
        expect_json: bool = False,
        # Memory thresholds (MB); None disables that check
        min_system_mb: int = 2500,
        min_vram_mb: int = 1500,
        cold_system_mb: int | None = 7000,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_input_chars = max_input_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.slow_death_s = slow_death_s
        self.thinking_off = thinking_off
        self.expect_json = expect_json
        self.min_system_mb = min_system_mb
        self.min_vram_mb = min_vram_mb
        self.cold_system_mb = cold_system_mb


class PreflightClient:
    """One configured connection to a local server.

    Usage:
        client = PreflightClient(ClientConfig(model="your-served-model-id"))
        try:
            text, usage = client.chat(system="...", user="...")
        except MemoryPressureError:
            defer_or_fallback()
        except LocalModelUnavailable as e:
            log_and_alert(e)
    """

    def __init__(self, config: ClientConfig | None = None):
        self.cfg = config or ClientConfig()
        # Warmth unknown at construction; omlx-style lazy servers keep the
        # model resident after any prior request (even from another client).
        # We probe lazily and cache; a live /models response means the server
        # has been serving, so treat "server reachable" as warm-after-probe.
        self._warm: bool | None = None

    # ── public ────────────────────────────────────────────────────────

    def preflight(self) -> dict:
        """Memory gate. Raises MemoryPressureError when starved."""
        if self._warm is None:
            # A reachable server that has our model listed is presumed warm:
            # lazy-loading servers load on first request anywhere, and a
            # running server with the model listed has almost always served
            # something (cron jobs, other clients) since its start.
            self._warm = self.health()
        ok, snap = memory_check(
            min_system_mb=self.cfg.min_system_mb,
            min_vram_mb=self.cfg.min_vram_mb,
            cold=not self._warm,
            cold_system_mb=self.cfg.cold_system_mb
            if self.cfg.cold_system_mb is not None
            else self.cfg.min_system_mb,
        )
        if not ok:
            raise MemoryPressureError(
                f"insufficient headroom: {snap} "
                f"(thresholds: system>={self.cfg.min_system_mb}MB "
                f"vram>={self.cfg.min_vram_mb}MB)"
            )
        return {"snapshot": str(snap), "limiting": snap.limiting_pool}

    def chat(self, system: str, user: str, **overrides) -> tuple[str, dict]:
        """Single disciplined call. Returns (text, usage-with-wall).

        Subject to the same five protections as Session.send():
        preflight check, truncation, thinking-off, starvation-aware retry,
        typed failures.
        """
        cfg = self.cfg
        if len(user) > cfg.max_input_chars:
            user = user[: cfg.max_input_chars] + "\n[truncated by llm-preflight]"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._do_request(messages, overrides)

    def _do_request(
        self,
        messages: list[dict],
        overrides: dict | None = None,
        apply_truncation: bool = True,
    ) -> tuple[str, dict]:
        """The single-shot request path shared by chat() and Session.

        Truncation, thinking-off, JSON mode, preflight, and the
        starvation-aware retry loop all live here exactly once, so any
        caller gets the same five protections.
        """
        overrides = overrides or {}
        cfg = self.cfg

        # Rule: hard input cap (apply to last user message by default —
        # caller can disable for Session which truncates at compaction time)
        if apply_truncation and messages and messages[-1].get("role") == "user":
            last = messages[-1]["content"]
            if len(last) > cfg.max_input_chars:
                messages = list(messages)
                messages[-1] = {**messages[-1], "content": last[: cfg.max_input_chars] + "\n[truncated by llm-preflight]"}

        self.preflight()

        body = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
            "temperature": overrides.get("temperature", cfg.temperature),
            "stream": False,
        }
        if cfg.thinking_off:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if overrides.get("expect_json", cfg.expect_json):
            body["response_format"] = {"type": "json_object"}
        if overrides.get("extra_body"):
            body.update(overrides["extra_body"])

        url = f"{cfg.base_url}/chat/completions"
        last_err: Exception | None = None

        for attempt in range(cfg.retries + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    data = json.loads(resp.read())
                wall = time.time() - t0
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
                text = strip_think(text)
                usage = dict(data.get("usage", {}))
                usage["wall_s"] = round(wall, 1)
                usage["attempt"] = attempt + 1
                self._warm = True  # model is resident now
                return text, usage
            except urllib.error.HTTPError as e:
                # Always read the error body — it's where servers name the
                # offending field. Silent 400s are debug nightmares.
                try:
                    body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    body = ""
                if 400 <= e.code < 500:
                    hint = ""
                    if cfg.thinking_off and ("chat_template_kwargs" in body or "extra" in body.lower() or "unknown" in body.lower()):
                        hint = " (server may reject chat_template_kwargs — try PreflightClient(thinking_off=False))"
                    raise LocalModelUnavailable(
                        f"HTTP {e.code}: {e.reason}{hint} | body: {body}"
                    ) from e
                # 5xx = transient server-side condition (model loading,
                # queue full, proxy hiccup) — same starvation-aware retry
                # policy as connection/timeout failures.
                wall = time.time() - t0
                last_err = e
                if wall > cfg.slow_death_s:
                    raise LocalModelUnavailable(
                        f"died after {wall:.0f}s (> {cfg.slow_death_s}s — "
                        f"HTTP {e.code}; treat as starvation, not retried)"
                    ) from e
                if attempt < cfg.retries:
                    time.sleep(2)
                    continue
                raise LocalModelUnavailable(f"HTTP {e.code} after retries: {e.reason}") from e
            except Exception as e:  # URLError, timeout, JSON decode
                wall = time.time() - t0
                last_err = e
                if wall > cfg.slow_death_s:
                    raise LocalModelUnavailable(
                        f"died after {wall:.0f}s (> {cfg.slow_death_s}s — "
                        f"treat as starvation, not retried)"
                    ) from e
                if attempt < cfg.retries:
                    time.sleep(2)
                    continue

        raise LocalModelUnavailable(f"failed after {cfg.retries + 1} attempts: {last_err}")

    def health(self) -> bool:
        """Cheap liveness check — server up and model discoverable."""
        try:
            with urllib.request.urlopen(
                f"{self.cfg.base_url}/models", timeout=5
            ) as r:
                return bool(json.loads(r.read()).get("data"))
        except Exception:
            return False
