"""EXPERIMENTAL (v0.2 preview): session manager for long interactive sessions.

The problem this addresses — distinct from preflight: preflight catches
"don't start while starved." It cannot catch "started fine, ran out of
memory 40 turns in." Long interactive sessions on constrained machines
degrade gradually as accumulated context grows the KV cache, then crash
mid-task. This module bounds that:

1. Context tracking via server-reported ``usage.prompt_tokens`` (never
   client-side tokenization — tokenizer mismatch makes that unreliable).
2. Compaction at a configurable fraction of the server's max context.
   Default strategy is TRUNCATION-WITH-ANCHORS, not summarize-and-replace:
   spending an LLM call to fix a resource problem, on the starved server,
   at the moment it is most starved, is backwards. Summarize is available
   as an opt-in strategy for users with headroom.
3. Turn-boundary checkpoints: the message list is persisted after each
   successful assistant turn. On crash, resume = reload messages, issue as
   a fresh request, let the server rebuild its own KV cache. We never
   touch server-side KV state — there is no portable API for it and
   pretending otherwise would tie this library to one engine.

Status: API may change without notice. The cron/single-shot client in
``llm_preflight.client`` is the stable, production-proven core; prefer it
for unattended work.
"""

from __future__ import annotations

import json


class SessionNotFoundError(FileNotFoundError):
    """Typed failure for Session.resume() — sid not on disk or corrupted."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from .client import ClientConfig, PreflightClient


@dataclass
class SessionConfig:
    compact_at_frac: float = 0.75      # of server max context
    anchor_turns: int = 2              # system + first N user/assistant pairs kept
    recent_window: int = 12            # sliding tail kept after anchors
    strategy: str = "truncate"         # "truncate" | "summarize"
    checkpoint_dir: Path = field(
        default_factory=lambda: Path.home() / ".llm-preflight" / "sessions"
    )
    checkpoint_every: int = 1          # persist every N successful turns


@dataclass
class SessionState:
    messages: list[dict] = field(default_factory=list)
    last_prompt_tokens: int = 0
    server_max_context: int | None = None
    compactions: int = 0
    turns: int = 0


class Session:
    """Stateful conversation wrapper with compaction + crash-resume."""

    def __init__(
        self,
        client_config: ClientConfig,
        session_config: SessionConfig | None = None,
        session_id: str | None = None,
    ):
        self.client = PreflightClient(client_config)
        self.cfg = session_config or SessionConfig()
        self.sid = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.state = SessionState()
        self._ckpt_path = self.cfg.checkpoint_dir / f"{self.sid}.json"

    # ── conversation ──────────────────────────────────────────────────

    def seed(self, system: str, first_user: str | None = None):
        self.state.messages = [{"role": "system", "content": system}]
        if first_user is not None:
            self.state.messages.append({"role": "user", "content": first_user})

    def send(self, user_text: str) -> tuple[str, dict]:
        """Append user turn, get assistant reply, track + compact + checkpoint."""
        self.state.messages.append({"role": "user", "content": user_text})
        self._maybe_compact_pre()

        msgs = list(self.state.messages)
        body_user = msgs[-1]["content"]
        # Use the client for the call but with full history: bypass its
        # single-shot shape by calling chat() with assembled transcript.
        text, usage = self._chat_with_history(msgs)

        self.state.messages.append({"role": "assistant", "content": text})
        self.state.turns += 1
        self.state.last_prompt_tokens = usage.get("prompt_tokens", 0) or 0
        if self.state.server_max_context is None:
            self.state.server_max_context = usage.get("server_max_context")
        # checkpoint_every=0 disables checkpointing entirely (no modulo)
        if self.cfg.checkpoint_every and self.state.turns % self.cfg.checkpoint_every == 0:
            self.checkpoint()
        self._maybe_compact_post()
        return text, usage

    # ── compaction ────────────────────────────────────────────────────

    def _compact_threshold_tokens(self) -> int | None:
        if not self.state.server_max_context:
            return None
        return int(self.state.server_max_context * self.cfg.compact_at_frac)

    def _maybe_compact_pre(self) -> None:
        thr = self._compact_threshold_tokens()
        if thr and self.state.last_prompt_tokens >= thr:
            self.compact()

    def _maybe_compact_post(self) -> None:
        # next call's prefill = last reported prompt tokens + new material;
        # compacting post-turn gives the next turn room.
        self._maybe_compact_pre()

    def compact(self) -> dict:
        """Apply the configured strategy. Returns a summary dict."""
        msgs = self.state.messages
        if len(msgs) <= self.cfg.anchor_turns * 2 + self.cfg.recent_window:
            return {"compacted": False, "reason": "below window"}

        system = msgs[0] if msgs and msgs[0]["role"] == "system" else None
        body = msgs[1:] if system else msgs
        anchors = body[: self.cfg.anchor_turns * 2]
        recent = body[-self.cfg.recent_window :]
        dropped = len(body) - len(anchors) - len(recent)

        if self.cfg.strategy == "summarize":
            transcript = "\n\n".join(
                f"[{m['role']}] {m['content'][:2000]}" for m in body
            )
            summary, _ = self.client.chat(
                system=(
                    "Summarize this conversation segment for continuity. "
                    "Keep decisions, facts, and open questions. <=300 words."
                ),
                user=transcript,
                max_tokens=512,
            )
            kept = anchors + [
                {"role": "user", "content": f"[Earlier conversation summary]\n{summary}"},
                {"role": "assistant", "content": "[acknowledged summary]"},
            ] + recent
        else:  # truncate (default)
            kept = anchors + [
                {"role": "user", "content": f"[{dropped} earlier turns truncated by llm-preflight]"},
                {"role": "assistant", "content": "[acknowledged]"},
            ] + recent

        new_msgs = ([system] if system else []) + kept
        self.state.messages = new_msgs
        self.state.compactions += 1
        return {"compacted": True, "dropped": dropped, "strategy": self.cfg.strategy}

    # ── checkpointing / resume ────────────────────────────────────────

    def checkpoint(self) -> Path:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.sid,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "state": {
                "messages": self.state.messages,
                "last_prompt_tokens": self.state.last_prompt_tokens,
                "server_max_context": self.state.server_max_context,
                "compactions": self.state.compactions,
                "turns": self.state.turns,
            },
        }
        tmp = self._ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self._ckpt_path)  # atomic
        return self._ckpt_path

    @classmethod
    def resume(
        cls,
        client_config: ClientConfig,
        session_id: str,
        session_config: SessionConfig | None = None,
    ) -> "Session":
        sess = cls(client_config, session_config, session_id=session_id)
        if not sess._ckpt_path.exists():
            raise SessionNotFoundError(
                f"no checkpoint for session_id={session_id!r} at {sess._ckpt_path}. "
                f"Pass the id returned by sess.sid after the last checkpoint()."
            )
        try:
            payload = json.loads(sess._ckpt_path.read_text())
            st = payload["state"]
            sess.state = SessionState(
                messages=st["messages"],
                last_prompt_tokens=st.get("last_prompt_tokens", 0),
                server_max_context=st.get("server_max_context"),
                compactions=st.get("compactions", 0),
                turns=st.get("turns", 0),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise SessionNotFoundError(
                f"checkpoint at {sess._ckpt_path} is corrupted "
                f"({type(e).__name__}: {e}). Delete the file or restore from backup."
            ) from e
        return sess

    # ── internals ─────────────────────────────────────────────────────

    def _chat_with_history(self, messages: list[dict]) -> tuple[str, dict]:
        """Send the full conversation through the shared request path.

        Routing through PreflightClient._do_request means Session gets the
        same five protections as a single chat() call: preflight check,
        truncation (Session manages its own compaction, so we disable
        the input-cap here), thinking-off, starvation-aware retry, and
        typed failures. Prior to this, Session silently bypassed all of
        them — users who set ClientConfig(max_input_chars=, slow_death_s=,
        retries=) expected those values to govern Session turns too.
        """
        return self.client._do_request(messages, apply_truncation=False)
