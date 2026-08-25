# llm-preflight

**Fail-safe client discipline for local LLMs on memory-constrained machines.**

Small local models fail *unattended* and *long-session* work in predictable, preventable ways: OOM kills mid-job, swap-thrash that turns a 20-second call into a 5-minute timeout, hidden reasoning tokens silently eating your budget, crashed sessions losing everything. This library is the client-side layer that catches each failure at the right seam — with any OpenAI-compatible server (omlx, llama.cpp, LM Studio, Ollama, FreeToken, vLLM).

> The model stays small. The system stops being fragile.

## Why this exists

The pain is real and documented in the wild — e.g. [openclaw#65551](https://github.com/openclaw/openclaw/issues/65551): *"Local MLX/LM Studio models get terminated on RAM pressure + no graceful handling"* in cron jobs. Local LLM reliability isn't a model problem; it's an **operational** problem, and nobody ships the operational layer.

Everything in this library was built to run real production cron jobs on a base M4 Mac Mini (24 GB), then generalized. The measured numbers behind every default are in [docs/MEASURED.md](docs/MEASURED.md).

## The five protections

| # | Protection | What it prevents |
|---|---|---|
| 1 | **Preflight memory check** — measures the pool your model actually allocates from (unified RAM on Apple Silicon; VRAM + RAM independently on discrete-GPU systems, gating on whichever is tighter) | Swap-thrash: the silent failure where a starved call burns minutes then times out, and a naive retry burns them again |
| 2 | **Starvation-aware retry** — attempts that die after `slow_death_s` are *never* retried | Doubling the cost of a doomed call |
| 3 | **Thinking-mode-off** — `chat_template_kwargs: {"enable_thinking": false}` by default | The hidden-reasoning tax: **4.7× more tokens, 4.1× slower** on tasks that don't need it (measured; prompt-level `/no_think` begging does *not* work) |
| 4 | **Token budgets** — hard input cap, bounded output | 60k-token prompts into a server that takes minutes to prefill |
| 5 | **Typed failures** — `MemoryPressureError` (defer, don't retry) vs `LocalModelUnavailable` (alert/fallback) | Silent garbage delivery |

## Quick start

```bash
pip install llm-preflight        # stdlib-only core; zero dependencies
```

```python
from llm_preflight import PreflightClient, ClientConfig, MemoryPressureError

client = PreflightClient(ClientConfig(
    base_url="http://127.0.0.1:8000/v1",
    model="your-served-model-id",  # from: curl $BASE/v1/models
))

try:
    text, usage = client.chat(
        system="You are a concise summarizer. Output only what is asked.",
        user=long_input_text,
    )
except MemoryPressureError:
    defer_or_fallback()   # RAM/VRAM starved — do NOT retry now
```

Or configure per-hardware with a TOML file:

```python
from llm_preflight.config import load_config
client = PreflightClient(load_config())  # reads ./llm-preflight.toml
```

```toml
# llm-preflight.toml
[server]
base_url = "http://127.0.0.1:8000/v1"
model = "your-served-model-id"   # exact id from your server's /v1/models

[memory]
min_system_mb = 2500
min_vram_mb = 1500        # gates independently on NVIDIA GPUs (pynvml, optional)
cold_system_mb = 7000     # extra headroom when the model isn't loaded yet

[retry]
slow_death_s = 90.0       # raise this for interactive long generations
```

## Long interactive sessions (experimental)

The distinct problem: preflight catches *"don't start while starved"*, not *"started fine, ran out of memory 40 turns in."* Long sessions die from context growth — exactly what happens running 35B-class MoE models on gaming GPUs. The experimental session manager bounds it:

```python
from llm_preflight.session import Session, SessionConfig

sess = Session(client_config, SessionConfig(
    compact_at_frac=0.75,   # compact at 75% of server max context
    strategy="truncate",    # anchors + sliding window (see docs for why
                            # summarize-and-replace is opt-in, not default)
))
sess.seed("You are a coding assistant.")
reply, usage = sess.send("Explain this error: ...")

# Crash? Resume from the last turn-boundary checkpoint:
sess2 = Session.resume(client_config, session_id=sess.sid, session_config=SessionConfig())
```

Checkpoints are the portable seam: we persist the **message list** at turn boundaries and let the server rebuild its own KV cache on resume. We never touch engine-internal KV state — there is no portable API for it, and pretending otherwise couples you to one engine.

⚠️ The session API is experimental (v0.2 preview) and may change. The single-shot client is the stable, production-proven core.

## Check your server first

"OpenAI-compatible" servers vary in what they actually implement (usage fields, thinking-mode kwargs, real context caps). The probe answers the three questions our design depends on:

```bash
llm-preflight-probe [base_url] [model]
# or: python -m llm_preflight.probe http://127.0.0.1:8000/v1
```

## What this library does NOT fix

Honest limits (see [docs/MEASURED.md](docs/MEASURED.md) for the full accounting): this fixes *availability* and *format* failures — crashes, timeouts, malformed output, memory exhaustion. It does nothing for *correctness* failures. A 9B/4-bit model will still confidently hallucinate facts inside perfectly valid JSON. Validation catches syntax, not truth. Route high-stakes calls to bigger models or the cloud, and keep the script owning the facts while the model owns the formatting.

## Development

```bash
git clone https://github.com/manulaggarwal/llm-preflight
cd llm-preflight
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q           # unit suite (no server needed)
.venv/bin/python tests/integration_live.py    # against your running server
```

The unit suite monkeypatches platform memory readers — it passes identically on macOS, Linux, and Windows CI without depending on the host's RAM state.

## License

MIT
