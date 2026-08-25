#!/usr/bin/env python3
"""
Capability probe — run this FIRST against any OpenAI-compatible
local server (FreeToken, omlx, llama.cpp, LM Studio, Ollama) before relying
on llm-preflight's session features.

Answers the three questions the design depends on, empirically:
  1. Does the server populate usage.prompt_tokens?  (auto-compact signal)
  2. Does it honor chat_template_kwargs.enable_thinking? (thinking-off)
  3. What max context does it actually enforce vs advertise?

Usage:  python -m llm_preflight.probe [base_url] [model]   (or: llm-preflight-probe [base_url] [model])
Default: http://127.0.0.1:8000/v1  (FreeToken default port)
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else None


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def chat(body, timeout=300):
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    import time
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    import sys as _sys
    if any(a in _sys.argv for a in ("--help", "-h")):
        print("Usage: llm-preflight-probe [BASE_URL] [MODEL]")
        print("Default base: http://127.0.0.1:8000/v1")
        print("Probes an OpenAI-compatible local server for usage/thinking/streaming support.")
        return 0
    print(f"Probing {BASE}\n")

    # 0. Model list + advertised context
    model = MODEL
    try:
        models = get("/models")["data"]
        print(f"models: {[m['id'] for m in models][:5]}")
        if model is None and models:
            model = models[0]["id"]
        for m in models:
            if m["id"] == model:
                print(f"  advertised max_model_len: {m.get('max_model_len', 'NOT REPORTED')}")
    except Exception as e:
        print(f"✗ /models failed: {e}")
        return 1

    # 1. usage.prompt_tokens present?
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 10,
        "temperature": 0,
    }
    try:
        resp, wall = chat(body)
        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens")
        print(f"\n1. usage.prompt_tokens: {'✅ ' + str(pt) if pt else '❌ MISSING'}")
        print(f"   usage keys present: {sorted(usage.keys())}")
        print(f"   wall: {wall:.1f}s")
    except Exception as e:
        print(f"\n1. ✗ chat failed: {e}")
        return 1

    # 2. chat_template_kwargs honored?
    body2 = dict(body)
    body2["chat_template_kwargs"] = {"enable_thinking": False}
    body2["max_tokens"] = 200
    body2["messages"] = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    try:
        resp2, _ = chat(body2)
        content = resp2["choices"][0]["message"].get("content", "") or ""
        finish = resp2["choices"][0].get("finish_reason")
        has_think = "<think>" in content or "reasoning_content" in resp2["choices"][0]["message"]
        ct = resp2.get("usage", {}).get("completion_tokens", "?")
        print(f"\n2. thinking-off: {'✅ no think block' if not has_think else '⚠️ think content present'}")
        print(f"   finish_reason: {finish}, completion_tokens: {ct}")
        print(f"   content[:80]: {content[:80]!r}")
        if ct and isinstance(ct, int) and ct < 30:
            print("   → looks efficient (short answer, no hidden reasoning)")
        else:
            print("   → possibly generating hidden reasoning despite kwarg — verify manually")
    except Exception as e:
        print(f"\n2. ✗ chat_template_kwargs call failed (server may reject unknown field): {e}")

    # 3. Streaming usage (many edge servers omit usage in stream mode)
    body3 = dict(body)
    body3["stream"] = True
    body3["stream_options"] = {"include_usage": True}
    try:
        req = urllib.request.Request(
            BASE + "/chat/completions",
            data=json.dumps(body3).encode(),
            headers={"Content-Type": "application/json"},
        )
        usage_in_stream = False
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: ") and "usage" in line and "[DONE]" not in line:
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage_in_stream = True
        print(f"\n3. streaming usage: {'✅ present' if usage_in_stream else '❌ absent (session tracker must poll non-stream or estimate)'}")
    except Exception as e:
        print(f"\n3. ⚠️ stream probe failed: {e}")

    print("\nDone. Share this output — it decides the session-manager design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
