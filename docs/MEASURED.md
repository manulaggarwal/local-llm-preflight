# MEASURED.md — every default in this library, and the data behind it

This library's defaults are not vibes. They come from running real cron jobs
(e-commerce store briefings, RSS digests, health checks) on a base M4 Mac Mini
(24 GB unified, 120 GB/s bandwidth) against omlx serving Ornith-1.5-9B-MLX-4bit,
plus a claude-consulted architecture review. Reproduce, then adjust.

## Table 1: prefill is the enemy (base M4, warm model)

| Prompt tokens | Wall time | Notes |
|---|---|---|
| ~500 | ~6s | decode-dominated |
| ~2k | ~13s | |
| ~8k | ~40s | prefill starts dominating |
| ~32k | ~138s | **2+ minutes — avoid** |

→ Default `max_input_chars = 96_000` (~24k tokens): the last point where
wall time stays sane on constrained hardware.

## Table 2: the hidden thinking tax

Same task (5-item feed → JSON summary), same model, same server:

| Configuration | Completion tokens | Wall time |
|---|---|---|
| Thinking on (default server behavior) | 602 | 30.0s |
| `chat_template_kwargs: {"enable_thinking": false}` | 128 | 7.3s |
| Prompt-level "/no_think" begging | ~550 | 27.7s |

**4.7× fewer tokens, 4.1× faster — and only the chat-template kwarg works.**
Verified twice through the production cron path (706→172 tokens end-to-end).

→ Default `thinking_off = True`, applied via `chat_template_kwargs`.
Note: servers vary in honoring this field — that's what the probe checks.

## Table 3: memory accounting gotchas (the false-fire story)

The macOS preflight originally counted only `free + purgeable`. Right after
a model load, macOS shows nearly all memory as *inactive* — which IS
reclaimable. Result: the check read "624 MB free" on a healthy system and
blocked every call.

| Counted pools | Reading on healthy warm system | Verdict |
|---|---|---|
| free + purgeable | 624 MB | ❌ false fire |
| free + purgeable + **inactive** | 9.2 GB | ✅ correct |

→ macOS reader counts all three. Linux uses `MemAvailable` (the kernel's
own reclaimable estimate). Windows uses `GlobalMemoryStatusex.ullAvailPhys`.

## Table 4: why starvation-aware retry exists

A starved local server doesn't fail fast. It thrashes: a 20s call takes
minutes, then times out. A naive retry re-runs the identical doomed call.

| Scenario | Naive client | llm-preflight |
|---|---|---|
| Server starved, 300s timeout | 300s burn × (1 + retries) | preflight blocks before the call: ~0s |
| Call dies at 120s under load | retry burns another 300s | `>90s death = never retry`: one attempt, typed error |

→ Default `slow_death_s = 90`. **For interactive long generations, raise it**
(a healthy 2k-token generation at 15 tok/s runs 130s+ legitimately).

## Failure modes NOT fixed by any of this

From the architecture review, worth keeping in the README's spirit of
honesty: all of this engineering fixes *availability* and *format* — not
*correctness*. A 9B/4-bit model produces wrong facts inside valid JSON;
quantization costs ~1-3% perplexity on top of the base model's ceiling, and
lands hardest on precise extraction and numeric reasoning. Mitigations that
actually work: route high-stakes calls up (bigger local model or cloud),
and spot-check outputs against ground truth where it matters.

## Session degradation anatomy (the Windows/35B case)

Long interactive sessions on pooled-VRAM engines (FreeToken-style):
every turn grows the KV cache; around turn 30-50 the working set crosses
from VRAM into system RAM; cross-boundary access is the "degrades" phase;
RAM-side thrash is the "crashes mid-task" phase. MoE doesn't change this —
all experts stay resident regardless of how few are active per token
(active params affect *compute*, not *residency*).

→ The session manager's job: compact at 75% of server-reported max context,
checkpoint at turn boundaries, resume by re-issuing the message list.
