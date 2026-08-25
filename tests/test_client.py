"""Tests for the disciplined client: budgets, retry policy, thinking-off."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_preflight.client import (
    ClientConfig,
    LocalModelUnavailable,
    MemoryPressureError,
    PreflightClient,
)


# ── tiny fake server ─────────────────────────────────────────────────

class _Fake:
    """Configurable fake OpenAI-compatible server."""

    def __init__(self):
        self.requests: list[dict] = []
        self.respond: dict = {}          # path -> callable(body) -> (status, payload)
        self.delay_s = 0.0

    handler_cls = None


def make_server(fake: _Fake):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            fake.requests.append({"path": self.path, "body": body})
            if fake.delay_s:
                import time
                time.sleep(fake.delay_s)
            handler = fake.respond.get(self.path) or fake.respond.get(
                self.path.rsplit("/v1", 1)[-1] if "/v1" in self.path else self.path
            )
            if handler:
                status, payload = handler(body)
            else:
                status, payload = 200, {"choices": [{"message": {"content": "ok"}}]}
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            p = self.path.rsplit("/v1", 1)[-1] if "/v1" in self.path else self.path
            if p == "/models":
                out = json.dumps({"data": [{"id": "test-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def fake_server():
    fake = _Fake()
    srv = make_server(fake)
    yield fake, f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def cfg_for(url, **kw):
    # Tests disable the memory gates entirely: unit tests must not depend
    # on the host machine's RAM state at run time.
    kw.setdefault("min_system_mb", 0)
    kw.setdefault("min_vram_mb", 0)
    kw.setdefault("cold_system_mb", None)
    return ClientConfig(base_url=url, model="test-model", **kw)


# ── tests ────────────────────────────────────────────────────────────

class TestBudgets:
    def test_input_cap_applied(self, fake_server):
        fake, url = fake_server
        client = PreflightClient(cfg_for(url, max_input_chars=1000, cold_system_mb=None))
        client.chat(system="s", user="x" * 5000)
        sent = fake.requests[-1]["body"]["messages"][1]["content"]
        assert len(sent) <= 1000 + len("\n[truncated by llm-preflight]")
        assert sent.endswith("[truncated by llm-preflight]")

    def test_thinking_off_sent_by_default(self, fake_server):
        fake, url = fake_server
        client = PreflightClient(cfg_for(url))
        client.chat(system="s", user="hi")
        assert fake.requests[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_flag_can_be_disabled(self, fake_server):
        fake, url = fake_server
        client = PreflightClient(cfg_for(url, thinking_off=False))
        client.chat(system="s", user="hi")
        assert "chat_template_kwargs" not in fake.requests[-1]["body"]

    def test_think_block_stripped_from_output(self, fake_server):
        fake, url = fake_server
        fake.respond["/chat/completions"] = lambda b: (200, {
            "choices": [{"message": {"content": "<think>internal</think>The answer."}}],
            "usage": {"prompt_tokens": 5},
        })
        client = PreflightClient(cfg_for(url))
        text, _ = client.chat(system="s", user="hi")
        assert text == "The answer."


class TestRetryPolicy:
    def test_fast_failure_retried_once(self, fake_server, monkeypatch):
        fake, url = fake_server
        calls = {"n": 0}

        def flaky(body):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("connection refused")
            return 200, {"choices": [{"message": {"content": "recovered"}}]}

        # URLError in handler -> 500; simulate fast failure differently
        class Boom(Exception):
            pass

        monkeypatch.setattr(
            "llm_preflight.client.urllib.request.urlopen",
            _flaky_urlopen(flaky, url),
        )
        client = PreflightClient(cfg_for(url, retries=1))
        text, usage = client.chat(system="s", user="hi")
        assert text == "recovered"
        assert calls["n"] == 2

    def test_slow_death_never_retried(self, fake_server, monkeypatch):
        fake, url = fake_server

        class SlowDeath(Exception):
            pass

        def slow(body):
            import time
            time.sleep(0.3)
            raise SlowDeath("starved")

        monkeypatch.setattr(
            "llm_preflight.client.urllib.request.urlopen",
            _flaky_urlopen(slow, url, wall_scale=1000),  # scale 0.3s -> reported 300s
        )
        client = PreflightClient(cfg_for(url, retries=3, slow_death_s=90))
        # Pre-set warmth so preflight does not add a health-probe call
        client._warm = True
        with pytest.raises(LocalModelUnavailable) as ei:
            client.chat(system="s", user="hi")
        assert "starvation" in str(ei.value)
        # exactly one CHAT attempt — no retries despite retries=3
        assert slow.calls == 1

    def test_http_4xx_not_retried(self, fake_server):
        fake, url = fake_server
        fake.respond["/chat/completions"] = lambda b: (404, {"error": "model not found"})
        client = PreflightClient(cfg_for(url, retries=5))
        with pytest.raises(LocalModelUnavailable):
            client.chat(system="s", user="hi")
        assert len(fake.requests) == 1


class TestPreflight:
    def test_memory_pressure_raises_typed_error(self, fake_server, monkeypatch):
        fake, url = fake_server
        from llm_preflight import memory as M

        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=500, vram_mb=None, platform="Darwin"))
        client = PreflightClient(cfg_for(url, min_system_mb=2500, cold_system_mb=None))
        with pytest.raises(MemoryPressureError):
            client.chat(system="s", user="hi")
        # crucially: no request was ever sent
        assert fake.requests == []


class TestHealth:
    def test_health_true_when_models_listed(self, fake_server):
        fake, url = fake_server
        client = PreflightClient(cfg_for(url))
        assert client.health() is True

    def test_health_false_when_down(self):
        client = PreflightClient(cfg_for("http://127.0.0.1:1/v1"))
        assert client.health() is False


# ── helper: patched urlopen with controllable behavior + wall scaling ─

def _flaky_urlopen(handler, base_url, wall_scale=1):
    class Resp:
        def __init__(self, payload):
            import io
            self._f = io.BytesIO(json.dumps(payload).encode())

        def read(self):
            return self._f.read()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        result = handler(body)
        if isinstance(result, tuple):
            status, payload = result
            if status >= 400:
                raise urllib.error.HTTPError(req.full_url, status, "err", {}, None)
            return Resp(payload)
        raise result  # exception instance

    # wall scaling: wrap time in the client module? simpler — monkeypatch time
    import llm_preflight.client as C
    orig_time = C.time.time

    class FakeTime:
        def time(self):
            return orig_time() * wall_scale

    if wall_scale != 1:
        C.time = FakeTime()
    handler.calls = getattr(handler, "calls", 0) + 0

    def counting(body):
        handler.calls = getattr(handler, "calls", 0) + 1
        return handler(body)

    # rebind so counting is used
    def urlopen_counted(req, timeout=None):
        handler.calls = getattr(handler, "calls", 0) + 1
        body = json.loads(req.data)
        result = handler.__wrapped__(body) if hasattr(handler, "__wrapped__") else handler(body)
        if isinstance(result, tuple):
            status, payload = result
            if status >= 400:
                raise urllib.error.HTTPError(req.full_url, status, "err", {}, None)
            return Resp(payload)
        raise result

    return urlopen_counted
