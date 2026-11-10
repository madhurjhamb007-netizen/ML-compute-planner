"""Shared deterministic fixtures for tests.

Hardware fixtures are synthetic test doubles with clearly labelled values. They
let estimation tests run deterministically without depending on the machine that
happens to execute the suite. Tests that validate real detection live in
``test_hardware.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas import CPUInfo, GPUInfo, HardwareInfo, RAMInfo


def make_cpu(physical: int = 8, logical: int = 16) -> CPUInfo:
    return CPUInfo(
        name="Test CPU 8C/16T",
        architecture="x86_64",
        physical_cores=physical,
        logical_threads=logical,
        current_frequency_mhz=2400.0,
        max_frequency_mhz=4200.0,
        utilization_pct=12.5,
        per_core_utilization_pct=[
            round(5.0 + index * 0.5, 2) for index in range(logical)
        ],
        recommended_threads=max(1, physical - 1),
        estimated_fp32_gflops=537.6,
        notes=["Synthetic hardware fixture for deterministic tests."],
    )


def make_ram(total_mb: float = 32768.0) -> RAMInfo:
    available = total_mb * 0.75
    return RAMInfo(
        total_mb=total_mb,
        available_mb=available,
        used_mb=total_mb - available,
        utilization_pct=25.0,
    )


def make_gpu(capability: str = "8.6") -> GPUInfo:
    return GPUInfo(
        index=0,
        name="Test GPU 24GB",
        total_memory_mb=24576.0,
        used_memory_mb=1024.0,
        free_memory_mb=23552.0,
        utilization_pct=4.0,
        memory_utilization_pct=2.0,
        compute_capability=capability,
        multiprocessors=84,
        clock_mhz=1770.0,
        temperature_c=45.0,
        power_w=60.0,
        driver_version="555.99",
        cuda_available=True,
        source="pytest fixture",
        notes=["Synthetic GPU fixture; not a real device."],
    )


def make_hardware(
    *,
    gpu: GPUInfo | None = None,
    cuda_available: bool = False,
) -> HardwareInfo:
    return HardwareInfo(
        hostname="test-host",
        os="Windows 11 (test)",
        python_version="3.14.0",
        detected_at=datetime.now(timezone.utc),
        detection_seconds=0.01,
        cpu=make_cpu(),
        ram=make_ram(),
        gpus=[gpu] if gpu is not None else [],
        cuda_available=cuda_available,
        torch_available=True,
        torch_version="2.15.0+cpu",
        torch_cuda_version="12.4" if cuda_available else None,
        nvidia_driver_version="555.99" if gpu is not None else None,
        warnings=[],
    )


@pytest.fixture()
def cpu_only_hardware() -> HardwareInfo:
    return make_hardware()


@pytest.fixture()
def gpu_hardware() -> HardwareInfo:
    return make_hardware(gpu=make_gpu("8.6"), cuda_available=True)


@pytest.fixture()
def pre_ampere_gpu_hardware(gpu_hardware: HardwareInfo) -> HardwareInfo:
    gpu = gpu_hardware.gpus[0].model_copy(update={"compute_capability": "7.5"})
    return gpu_hardware.model_copy(
        update={
            "gpus": [gpu],
            "torch_cuda_version": "11.8",
        }
    )
