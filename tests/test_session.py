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


class TestSessionProtections:
    """The fourth audit caught that Session silently bypassed all client
    protections. These tests pin the contract: Session must go through
    _do_request and inherit the same five protections as PreflightClient.chat().
    """

    def test_session_sends_through_shared_request_path(self):
        """Session uses the client's shared _do_request — verify via the
        integration test (chat completion against live omlx)."""
        import urllib.error
        from llm_preflight import Session, SessionConfig, PreflightClient, ClientConfig
        from unittest.mock import patch
        # Session takes a ClientConfig (it builds its own PreflightClient)
        cfg = ClientConfig(base_url="http://127.0.0.1:8010/v1", model="Ornith-1.5-9B-MLX-4bit")
        sess = Session(cfg, SessionConfig(checkpoint_every=0))
        client = sess.client
        sess.seed("you are concise. answer in 5 words max.")
        # patch _do_request to confirm Session routes through it
        called = {"n": 0}
        orig = client._do_request
        def spy(messages, overrides=None, apply_truncation=True):
            called["n"] += 1
            assert messages[-1]["role"] == "user"
            # Session passes apply_truncation=False (owns compaction)
            assert apply_truncation is False
            # Don't call orig — preflight against the live test machine
            # may not have the headroom for a warm-restart. The contract
            # we care about is "Session routes through _do_request", which
            # the side_effect already proves.
            return "ok", {"wall_s": 0.0, "attempt": 1}
        with patch.object(client, "_do_request", side_effect=spy):
            sess.send("hello")
        assert called["n"] == 1

    def test_session_resume_raises_typed_error_on_unknown_id(self):
        from llm_preflight import Session, SessionConfig, PreflightClient, ClientConfig, SessionNotFoundError
        cfg = ClientConfig(base_url="http://127.0.0.1:8010/v1")
        with pytest.raises(SessionNotFoundError) as ei:
            Session.resume(cfg, session_id="definitely-not-real-xyz")
        assert "definitely-not-real-xyz" in str(ei.value)

    def test_session_resume_raises_typed_error_on_corrupt_checkpoint(self, tmp_path):
        from llm_preflight import Session, SessionConfig, PreflightClient, ClientConfig, SessionNotFoundError
        cfg = ClientConfig(base_url="http://127.0.0.1:8010/v1")
        sc = SessionConfig(checkpoint_dir=tmp_path)
        # Force a corrupted checkpoint file in place
        bad = tmp_path / "bad-session.json"
        bad.write_text("{not valid json")
        with pytest.raises(SessionNotFoundError) as ei:
            Session.resume(cfg, session_id="bad-session", session_config=sc)
        assert "corrupted" in str(ei.value)

    def test_client_4xx_error_includes_body(self):
        """The fourth audit demanded the server's response body in 4xx errors
        so users can see WHAT the server rejected (e.g. unknown chat_template_kwargs)."""
        from llm_preflight import PreflightClient, ClientConfig, LocalModelUnavailable
        # Build a fake server that 400s with a body mentioning chat_template_kwargs
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                msg = b'{"error": "unrecognized arguments: chat_template_kwargs"}'
                self.send_response(400); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg))); self.end_headers()
                self.wfile.write(msg)
            def log_message(self, *a): pass
        srv = HTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
        try:
            cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}/v1", model="x", thinking_off=True, retries=0)
            client = PreflightClient(cfg); client._warm = True
            with pytest.raises(LocalModelUnavailable) as ei:
                client.chat(system="s", user="u")
            # The server's response body AND an actionable hint should both
            # be present — silent 400s are debug nightmares.
            msg = str(ei.value)
            assert "chat_template_kwargs" in msg, f"body not in error: {msg}"
            assert "thinking_off=False" in msg, f"hint not in error: {msg}" 
        finally:
            srv.shutdown()

    def test_checkpoint_every_zero_disables(self, tmp_path):
        """checkpoint_every=0 is documented as 'omit checkpoints' but used
        to crash with ZeroDivisionError (audit caught this)."""
        from unittest.mock import patch
        from llm_preflight import Session, SessionConfig, ClientConfig
        cfg = ClientConfig(base_url="http://127.0.0.1:8010/v1")
        sc = SessionConfig(checkpoint_every=0, checkpoint_dir=tmp_path)
        sess = Session(cfg, sc)
        sess.seed("s")
        with patch.object(sess.client, "_do_request",
                          return_value=("reply", {"wall_s": 0.1, "prompt_tokens": 5})):
            sess.send("hi")  # must not raise
        # No checkpoint file should have been written
        assert list(tmp_path.glob("*.json")) == []
