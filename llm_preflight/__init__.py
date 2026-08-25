"""
llm-preflight
=============

Fail-safe client discipline for local LLMs on memory-constrained machines.

The problem: small local models on constrained systems fail *unattended*
work in predictable, preventable ways — OOM kills mid-job, swap-thrash that
turns a 20-second call into a 5-minute timeout, hidden reasoning tokens
eating the budget, malformed output silently delivered. This library is the
client-side discipline layer that catches each of those at the right seam.

Quick start:

    from llm_preflight import PreflightClient, ClientConfig, MemoryPressureError

    client = PreflightClient(ClientConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="your-served-model-id",
    ))
    try:
        text, usage = client.chat(
            system="You are a concise summarizer.",
            user=summarize_this_text,
        )
    except MemoryPressureError:
        ...  # defer the job or use a plain-text fallback — never retry now
"""

from .client import (
    ClientConfig,
    LocalModelUnavailable,
    MemoryPressureError,
    PreflightClient,
)
from .memory import MemorySnapshot, check, snapshot

__version__ = "0.1.0"
__all__ = [
    "ClientConfig",
    "PreflightClient",
    "MemoryPressureError",
    "LocalModelUnavailable",
    "MemorySnapshot",
    "snapshot",
    "check",
    "__version__",
]
