"""Hardware detection: real-fact consistency and failure fallbacks."""

from __future__ import annotations

import math

import psutil

from app.hardware import detector
from app.hardware.cpu import detect_cpu, detect_ram
from app.hardware.detector import detect_hardware, primary_gpu
from app.hardware.gpu import _merge_entries


def test_cpu_detection_matches_psutil_without_inventing_cores() -> None:
    cpu = detect_cpu()
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False)

    assert cpu.logical_threads == logical
    if physical is not None:
        assert cpu.physical_cores == physical
    assert 1 <= cpu.recommended_threads <= cpu.logical_threads
    assert cpu.estimated_fp32_gflops > 0
    assert 0.0 <= cpu.utilization_pct <= 100.0
    if cpu.per_core_utilization_pct:
        assert len(cpu.per_core_utilization_pct) == cpu.logical_threads
        assert all(0.0 <= value <= 100.0 for value in cpu.per_core_utilization_pct)


def test_ram_detection_is_internally_consistent() -> None:
    ram = detect_ram()
    assert ram.total_mb > 0
    assert 0 <= ram.available_mb <= ram.total_mb
    assert 0 <= ram.used_mb <= ram.total_mb
    assert 0 <= ram.utilization_pct <= 100


def test_full_snapshot_reports_only_consistent_gpu_facts() -> None:
    hardware = detect_hardware()
    assert hardware.hostname
    assert hardware.os
    assert hardware.cpu.physical_cores >= 1
    assert hardware.detection_seconds >= 0
    assert math.isfinite(hardware.detection_seconds)

    if hardware.cuda_available:
        assert hardware.torch_available
        assert hardware.gpus, "CUDA cannot be reported available without a visible GPU"

    for gpu in hardware.gpus:
        assert gpu.index >= 0
        assert gpu.name
        assert gpu.total_memory_mb > 0
        if gpu.used_memory_mb is not None:
            assert 0 <= gpu.used_memory_mb <= gpu.total_memory_mb
        if gpu.free_memory_mb is not None:
            assert 0 <= gpu.free_memory_mb <= gpu.total_memory_mb
        if gpu.utilization_pct is not None:
            assert 0 <= gpu.utilization_pct <= 100


def test_primary_gpu_returns_first_or_none(cpu_only_hardware, gpu_hardware) -> None:
    assert primary_gpu(cpu_only_hardware) is None
    assert primary_gpu(gpu_hardware) is gpu_hardware.gpus[0]


def test_detection_falls_back_when_every_backend_raises(monkeypatch) -> None:
    def fail() -> None:
        raise RuntimeError("forced detection failure")

    monkeypatch.setattr(detector, "detect_cpu", fail)
    monkeypatch.setattr(detector, "detect_ram", fail)
    monkeypatch.setattr(detector, "detect_gpus", fail)

    hardware = detector.detect_hardware()

    assert hardware.cpu.name == "Unknown CPU"
    assert hardware.cpu.physical_cores == 1
    assert hardware.cpu.logical_threads == 1
    assert hardware.ram.total_mb == 0
    assert hardware.gpus == []
    assert hardware.cuda_available is False
    assert hardware.torch_available is False
    assert any("CPU detection failed" in warning for warning in hardware.warnings)
    assert any("RAM detection failed" in warning for warning in hardware.warnings)
    assert any("GPU detection failed" in warning for warning in hardware.warnings)


def test_gpu_sources_merge_without_overwriting_richer_values() -> None:
    base = {
        0: {
            "index": 0,
            "name": "GPU A",
            "total_memory_mb": 8192.0,
            "utilization_pct": 37.0,
            "source": "nvidia-smi",
        }
    }
    _merge_entries(
        base,
        [
            {
                "index": 0,
                "name": "GPU A",
                "utilization_pct": 99.0,
                "compute_capability": "8.6",
                "source": "NVML",
            },
            {
                "index": 1,
                "name": "GPU B",
                "total_memory_mb": 16384.0,
                "source": "NVML",
            },
        ],
    )

    assert base[0]["utilization_pct"] == 37.0
    assert base[0]["compute_capability"] == "8.6"
    assert base[0]["source"] == "NVML+nvidia-smi"
    assert base[1]["name"] == "GPU B"
