#!/usr/bin/env python3
"""Integration test against a LIVE local server.

Skipped automatically when no server is reachable — CI runs the unit suite;
this file proves the library against reality (omlx on the dev machine,
FreeToken/llama.cpp/LM Studio on yours).

Run:  .venv/bin/python tests/integration_live.py [base_url]
"""
import json
import sys
import urllib.request

from llm_preflight import (
    ClientConfig,
    LocalModelUnavailable,
    MemoryPressureError,
    PreflightClient,
)

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010/v1"

passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


# 0. server reachable?
try:
    with urllib.request.urlopen(f"{BASE}/models", timeout=3) as r:
        model = json.loads(r.read())["data"][0]["id"]
except Exception as e:
    print(f"SKIP — no server at {BASE}: {e}")
    sys.exit(0)

print(f"Integration against {BASE} (model: {model})\n")

client = PreflightClient(ClientConfig(base_url=BASE, model=model))


def t_health():
    assert client.health() is True


def t_preflight_snap():
    info = client.preflight()
    assert "limiting" in info


def t_basic_chat():
    text, usage = client.chat(system="Answer with one word.", user="What is 2+2?")
    assert "4" in text, f"got {text!r}"
    assert usage.get("prompt_tokens", 0) > 0
    assert "wall_s" in usage


def t_thinking_off():
    text, usage = client.chat(
        system="Answer concisely.", user="Name the capital of France. One word."
    )
    assert "Paris" in text
    ct = usage.get("completion_tokens", 0)
    assert ct < 50, f"completion suspiciously large ({ct}) — thinking may be ON"


def t_input_truncation():
    text, usage = client.chat(
        system="Summarize in <=10 words.",
        user="x" * 200_000,  # way over default cap
        max_tokens=64,
    )
    pt = usage.get("prompt_tokens", 0)
    assert pt < 30_000, f"prompt_tokens={pt} — cap not applied?"


def t_json_mode():
    text, usage = client.chat(
        system='Return ONLY JSON: {"ok": true}',
        user='Return the JSON.',
        expect_json=True,
        max_tokens=64,
    )
    parsed = json.loads(text)
    assert parsed.get("ok") is True


def t_session_roundtrip():
    from llm_preflight.session import Session, SessionConfig
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        sess = Session(
            ClientConfig(base_url=BASE, model=model, max_tokens=64),
            SessionConfig(checkpoint_dir=Path(td)),
            session_id="integration",
        )
        sess.seed("You are terse.", "Remember the number 7.")
        # seed leaves [system, user]; send a turn to get an assistant reply
        text, _ = sess.send("What number did I ask you to remember? Just the digit.")
        sess.checkpoint()
        restored = Session.resume(
            ClientConfig(base_url=BASE, model=model, max_tokens=64), "integration",
            SessionConfig(checkpoint_dir=Path(td)),
        )
        assert len(restored.state.messages) == len(sess.state.messages)
        assert restored.state.turns == sess.state.turns


for name, fn in [
    ("health", t_health),
    ("preflight snapshot", t_preflight_snap),
    ("basic chat + usage", t_basic_chat),
    ("thinking-off (short answers)", t_thinking_off),
    ("input truncation cap", t_input_truncation),
    ("json mode", t_json_mode),
    ("session checkpoint/resume roundtrip", t_session_roundtrip),
]:
    check(name, fn)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
