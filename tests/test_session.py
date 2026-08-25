"""Tests for the experimental session manager: compaction + checkpoint/resume."""

import json

import pytest

from llm_preflight.client import ClientConfig
from llm_preflight.session import Session, SessionConfig


class TestCompaction:
    def _mk_session(self, tmp_path, n_turns=30, **kw):
        scfg = SessionConfig(checkpoint_dir=tmp_path / "sessions", **kw)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(n_turns):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        sess = Session(ClientConfig(model="m"), scfg, session_id="t1")
        sess.state.messages = msgs
        return sess

    def test_truncate_keeps_anchors_and_recent(self, tmp_path):
        sess = self._mk_session(tmp_path)
        result = sess.compact()
        assert result["compacted"] is True
        msgs = sess.state.messages
        assert msgs[0]["content"] == "sys"                       # system kept
        assert msgs[1]["content"] == "u0"                        # anchor kept
        assert msgs[-1]["content"] == "a29"                      # recent kept
        assert any("truncated by llm-preflight" in m["content"] for m in msgs)
        assert sess.state.compactions == 1

    def test_no_compact_below_window(self, tmp_path):
        sess = self._mk_session(tmp_path, n_turns=3)
        result = sess.compact()
        assert result["compacted"] is False

    def test_threshold_triggers_compact(self, tmp_path):
        sess = self._mk_session(tmp_path)
        sess.state.server_max_context = 10_000
        sess.state.last_prompt_tokens = 8_000  # > 0.75 * 10000
        sess._maybe_compact_pre()
        assert sess.state.compactions == 1

    def test_summarize_strategy_calls_llm(self, tmp_path, monkeypatch):
        sess = self._mk_session(tmp_path, strategy="summarize")
        captured = {}

        def fake_chat(system, user, **kw):
            captured["system"] = system
            return "SUMMARY", {}

        monkeypatch.setattr(sess.client, "chat", fake_chat)
        result = sess.compact()
        assert result["strategy"] == "summarize"
        assert any("SUMMARY" in m["content"] for m in sess.state.messages)
        assert "continuity" in captured["system"]


class TestCheckpoint:
    def test_checkpoint_and_resume_roundtrip(self, tmp_path):
        scfg = SessionConfig(checkpoint_dir=tmp_path / "s")
        sess = Session(ClientConfig(model="m"), scfg, session_id="rt")
        sess.seed("You are a test.", "hello")
        sess.state.messages.append({"role": "assistant", "content": "hi"})
        sess.state.turns = 1
        sess.state.last_prompt_tokens = 42
        path = sess.checkpoint()
        assert path.exists()

        restored = Session.resume(ClientConfig(model="m"), "rt", scfg)
        assert restored.state.messages == sess.state.messages
        assert restored.state.turns == 1
        assert restored.state.last_prompt_tokens == 42

    def test_checkpoint_is_atomic_json(self, tmp_path):
        import time
        scfg = SessionConfig(checkpoint_dir=tmp_path / "s")
        sess = Session(ClientConfig(model="m"), scfg, session_id="atomic")
        sess.seed("s", "u")
        p = sess.checkpoint()
        data = json.loads(p.read_text())
        assert data["session_id"] == "atomic"
        assert "saved_at" in data
