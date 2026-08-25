from __future__ import annotations

import os
"""Cross-platform memory headroom detection for local LLM inference.

The core insight this module implements: a memory check is only valid if it
measures the pool the model actually allocates from.

- macOS / Apple Silicon: unified memory — free + purgeable + inactive system
  RAM is the right signal. (Counting only "free" is wrong: right after a model
  loads, macOS shows almost everything as inactive, which IS reclaimable.)
- Windows / Linux + discrete GPU: weights and KV cache live primarily in
  VRAM. System RAM alone tells you little. We report BOTH pools and gate on
  whichever is more constrained. VRAM detection is NVIDIA-only (pynvml,
  optional) and explicitly reported as unavailable otherwise — never silently
  ignored.
- Integrated GPUs / pooled allocators (e.g. MoE serving engines that
  overflow experts from VRAM into system RAM): both pools matter; see above.

This module never raises on measurement failure — an unreadable metric
returns ``None`` and callers decide what that means for them.
"""

import platform
import subprocess
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Reclaimable headroom, in MB. ``None`` = could not measure."""

    system_mb: int | None
    vram_mb: int | None          # NVIDIA-only; None on AMD/Intel/macOS
    platform: str

    @property
    def limiting_pool(self) -> str:
        """Which pool is the binding constraint for gating decisions."""
        if self.vram_mb is not None and self.system_mb is not None:
            return "vram" if self.vram_mb < self.system_mb else "system"
        if self.vram_mb is not None:
            return "vram"
        return "system"

    @property
    def limiting_mb(self) -> int | None:
        return getattr(self, f"{self.limiting_pool}_mb", None)

    def __str__(self) -> str:  # pragma: no cover - debug aid
        s = f"system={self.system_mb}MB" if self.system_mb is not None else "system=?"
        v = f"vram={self.vram_mb}MB" if self.vram_mb is not None else "vram=n/a"
        return f"[{self.platform}] {s} {v} -> limit={self.limiting_pool}"


_PAGE_SIZE_FALLBACK = (os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 16384)  # portable; vm_stat usually announces the size itself


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


# ── macOS ────────────────────────────────────────────────────────────────

def _system_mb_macos() -> int | None:
    out = _run(["vm_stat"])
    if not out:
        return None
    page = _PAGE_SIZE_FALLBACK
    free = purgeable = inactive = 0
    for line in out.split("\n"):
        if "page size of" in line:
            try:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            except (ValueError, IndexError):
                pass
        if "Pages free" in line:
            free = int(line.split()[2].rstrip("."))
        elif "Pages purgeable" in line:
            purgeable = int(line.split()[2].rstrip("."))
        elif "Pages inactive" in line:
            inactive = int(line.split()[2].rstrip("."))
    return (free + purgeable + inactive) * page // (1024 * 1024)


# ── Windows ──────────────────────────────────────────────────────────────

def _system_mb_windows() -> int | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys) // (1024 * 1024)
    except Exception:
        pass
    # psutil fallback if present
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return None


# ── Linux ────────────────────────────────────────────────────────────────

def _system_mb_linux() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0])  # kB
            # MemAvailable is the kernel's own reclaimable estimate
            return info.get("MemAvailable", 0) // 1024 or None
    except Exception:
        return None


# ── NVIDIA VRAM (optional, cross-platform where the driver exists) ───────

def _vram_mb_nvidia() -> int | None:
    # Preferred: pynvml (in-process, no subprocess on the hot path)
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            total = free = 0
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                total += info.total
                free += info.free
            return int(free // (1024 * 1024)) if total else None
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    # Fallback: nvidia-smi text scrape
    out = _run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
    if out:
        try:
            return sum(int(x) for x in out.strip().split("\n") if x.strip())
        except ValueError:
            return None
    return None


# ── public API ───────────────────────────────────────────────────────────

def snapshot() -> MemorySnapshot:
    """Take a memory snapshot. Never raises; unmeasured pools are None."""
    system = platform.system()
    if system == "Darwin":
        sys_mb = _system_mb_macos()
    elif system == "Windows":
        sys_mb = _system_mb_windows()
    else:
        sys_mb = _system_mb_linux()
    return MemorySnapshot(
        system_mb=sys_mb,
        vram_mb=_vram_mb_nvidia(),
        platform=system,
    )


def check(
    min_system_mb: int = 2500,
    min_vram_mb: int = 1500,
    cold: bool = False,
    cold_system_mb: int = 7000,
) -> tuple[bool, MemorySnapshot]:
    """Gate decision for 'should we attempt local inference right now?'.

    Returns (ok, snapshot). ``ok=False`` means defer or fall back — do NOT
    sit and retry; the constraint will not clear in seconds.

    ``cold=True`` means the model is not yet resident and loading needs more
    headroom than steady-state inference (``cold_system_mb``). Callers that
    don't track warmth should pass ``cold=True`` on their first call after
    server start — PreflightClient does this for you.
    """
    snap = snapshot()
    limiting = snap.limiting_mb
    threshold = (
        min_vram_mb if snap.limiting_pool == "vram" else min_system_mb
    )
    if limiting is not None and limiting < threshold:
        return False, snap
    if cold and snap.system_mb is not None and snap.system_mb < cold_system_mb:
        return False, snap
    return True, snap
