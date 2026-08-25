#!/usr/bin/env python3
"""Example: a cron-safe daily digest using llm-preflight.

Pattern: the SCRIPT owns the facts (gathering, filtering, validation) and
the MODEL owns the formatting. That split is deliberate — it's what keeps
a small model's hallucination surface tiny. If the model is unavailable,
the script degrades to plain output instead of failing.

Reads LLM_PREFLIGHT_TOML if set, else ./llm-preflight.toml, else defaults.
"""

import json
import os
import sys

from llm_preflight import ClientConfig, MemoryPressureError, PreflightClient
from llm_preflight.config import load_config

# ── Step 1: gather facts (no LLM) — your data source goes here ────────

SAMPLE_ITEMS = [
    "v1.2 of a tool you use released: streaming support, memory fixes.",
    "A library you depend on is deprecated; migrate by year end.",
    "Benchmarks show local 9B models matching cloud quality on summaries.",
]


def gather():
    """Replace with your real data source (RSS, API, DB query...)."""
    return SAMPLE_ITEMS


def main():
    items = gather()
    if not items:
        return  # silent when nothing to do — cron etiquette

    # LLM_PREFLIGHT_TOML env var -> explicit path; else auto-discover
    cfg_path = os.environ.get("LLM_PREFLIGHT_TOML")  # None -> search defaults
    client = PreflightClient(load_config(cfg_path))

    feed = "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
    system = (
        "You format a daily digest. Output max 6 bullet lines, each one line: "
        "- item — why it matters (<=15 words). No preamble."
    )

    try:
        text, usage = client.chat(system=system, user=f"Items:\n{feed}")
    except MemoryPressureError:
        # Starved: degrade to plain output, never block the pipeline
        print("Digest (plain fallback — model deferred under memory pressure):")
        for t in items:
            print(f"- {t}")
        return
    except Exception as e:
        print(f"Digest (plain fallback — model unavailable: {e}):", file=sys.stderr)
        for t in items:
            print(f"- {t}")
        return

    print(text)
    print(f"\n— llm-preflight: {usage.get('prompt_tokens', '?')} in / "
          f"{usage.get('completion_tokens', '?')} out tok, "
          f"{usage.get('wall_s', '?')}s", file=sys.stderr)


if __name__ == "__main__":
    main()
