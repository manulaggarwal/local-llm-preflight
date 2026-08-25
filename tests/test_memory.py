"""Unit tests for the memory module — the part that must never lie.

Platform-dependent readers are monkeypatched so these tests run identically
on macOS CI, Windows, and Linux.
"""

import pytest

from llm_preflight import memory as M


class TestMemorySnapshot:
    def test_limiting_pool_prefers_vram_when_tighter(self):
        snap = M.MemorySnapshot(system_mb=8000, vram_mb=500, platform="Windows")
        assert snap.limiting_pool == "vram"
        assert snap.limiting_mb == 500

    def test_limiting_pool_system_when_no_vram(self):
        snap = M.MemorySnapshot(system_mb=1000, vram_mb=None, platform="Darwin")
        assert snap.limiting_pool == "system"
        assert snap.limiting_mb == 1000

    def test_vram_only(self):
        snap = M.MemorySnapshot(system_mb=None, vram_mb=2000, platform="Windows")
        assert snap.limiting_pool == "vram"


class TestCheck:
    def test_blocks_when_system_starved(self, monkeypatch):
        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=800, vram_mb=None, platform="Darwin"))
        ok, snap = M.check(min_system_mb=2500)
        assert ok is False
        assert snap.system_mb == 800

    def test_passes_when_healthy(self, monkeypatch):
        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=9000, vram_mb=None, platform="Darwin"))
        ok, _ = M.check(min_system_mb=2500)
        assert ok is True

    def test_vram_is_the_gate_on_gpu_systems(self, monkeypatch):
        # RAM plentiful, VRAM starved — must block (the Windows friend case)
        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=32000, vram_mb=400, platform="Windows"))
        ok, snap = M.check(min_system_mb=2500, min_vram_mb=1500)
        assert ok is False
        assert snap.limiting_pool == "vram"

    def test_cold_start_needs_more_headroom(self, monkeypatch):
        # 4000MB: fine warm (2500 threshold), blocked cold (7000 threshold)
        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=4000, vram_mb=None, platform="Darwin"))
        ok_warm, _ = M.check(min_system_mb=2500, cold=False)
        ok_cold, _ = M.check(min_system_mb=2500, cold=True, cold_system_mb=7000)
        assert ok_warm is True
        assert ok_cold is False

    def test_unmeasurable_never_blocks(self, monkeypatch):
        # If we cannot read memory, do not brick the caller's pipeline
        monkeypatch.setattr(M, "snapshot", lambda: M.MemorySnapshot(
            system_mb=None, vram_mb=None, platform="Unknown"))
        ok, _ = M.check()
        assert ok is True


class TestParsers:
    def test_macos_vm_stat_parse(self, monkeypatch):
        sample = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   1000.
Pages active:                                 2000.
Pages inactive:                               3000.
Pages speculative:                            100.
Pages wired down:                             500.
Pages purgeable:                              200.
"""
        monkeypatch.setattr(M, "_run", lambda cmd: sample if cmd == ["vm_stat"] else None)
        # (1000 + 200 + 3000) * 16384 / 1024^2 = 65.625 -> 65 MB
        assert M._system_mb_macos() == 65

    def test_linux_meminfo_parse(self, monkeypatch, tmp_path):
        monkeypatch.setattr("builtins.open", lambda p, *a, **k: __import__("io").StringIO(
            "MemTotal:       16000000 kB\nMemAvailable:    4000000 kB\n"
        ))
        assert M._system_mb_linux() == 4000000 // 1024

    def test_snapshot_never_raises(self):
        # On any real machine this must return SOMETHING without exploding
        snap = M.snapshot()
        assert isinstance(snap, M.MemorySnapshot)
        assert snap.platform in ("Darwin", "Windows", "Linux")
