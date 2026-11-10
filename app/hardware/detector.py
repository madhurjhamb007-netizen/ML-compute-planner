"""Aggregate a consistent hardware snapshot."""

from __future__ import annotations

import logging
import platform
import socket
import sys
import time
from datetime import datetime, timezone

from app.hardware.cpu import detect_cpu, detect_ram
from app.hardware.gpu import detect_gpus
from app.schemas import HardwareInfo

logger = logging.getLogger(__name__)


def detect_hardware() -> HardwareInfo:
    """Detect CPU, RAM, GPUs, CUDA, and PyTorch state.

    Detection is defensive: every subsystem has a fallback so the application can
    start and provide CPU-only estimates on machines without NVIDIA drivers.
    """
    started = time.perf_counter()
    warnings: list[str] = []

    try:
        cpu = detect_cpu()
    except Exception as exc:  # noqa: BLE001
        logger.exception("CPU detection failed")
        warnings.append(f"CPU detection failed: {exc}")
        from app.schemas import CPUInfo

        cpu = CPUInfo(
            name="Unknown CPU",
            architecture=platform.machine() or "unknown",
            physical_cores=1,
            logical_threads=1,
            utilization_pct=0.0,
            per_core_utilization_pct=[],
            recommended_threads=1,
            estimated_fp32_gflops=8.0,
            notes=["Fallback values used because CPU detection failed."],
        )

    try:
        ram = detect_ram()
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAM detection failed")
        warnings.append(f"RAM detection failed: {exc}")
        from app.schemas import RAMInfo

        ram = RAMInfo(total_mb=0.0, available_mb=0.0, used_mb=0.0, utilization_pct=0.0)

    try:
        gpus, gpu_meta = detect_gpus()
    except Exception as exc:  # noqa: BLE001
        logger.exception("GPU detection failed")
        warnings.append(f"GPU detection failed; continuing CPU-only: {exc}")
        gpus = []
        gpu_meta = {
            "torch_available": False,
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "nvidia_driver_version": None,
            "notes": [],
        }

    if gpus and not gpu_meta.get("cuda_available"):
        warnings.append(
            "A GPU was detected, but PyTorch cannot use CUDA. Install a CUDA-enabled "
            "PyTorch build matching the driver to benchmark on GPU."
        )
    if gpu_meta.get("notes"):
        warnings.extend(gpu_meta["notes"])
    if not gpu_meta.get("torch_available"):
        warnings.append(
            "PyTorch is not installed or cannot be imported; model benchmarks are unavailable."
        )

    snapshot = HardwareInfo(
        hostname=socket.gethostname(),
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python_version=sys.version.split()[0],
        detected_at=datetime.now(timezone.utc),
        detection_seconds=0.0,
        cpu=cpu,
        ram=ram,
        gpus=gpus,
        cuda_available=bool(gpu_meta.get("cuda_available")),
        torch_available=bool(gpu_meta.get("torch_available")),
        torch_version=gpu_meta.get("torch_version"),
        torch_cuda_version=gpu_meta.get("torch_cuda_version"),
        nvidia_driver_version=gpu_meta.get("nvidia_driver_version"),
        warnings=list(dict.fromkeys(warnings)),
    )
    elapsed = time.perf_counter() - started
    return snapshot.model_copy(update={"detection_seconds": round(elapsed, 3)})


def primary_gpu(hardware: HardwareInfo):  # type: ignore[no-untyped-def]
    """Return the first GPU, or None. Small helper kept lazy to avoid imports."""
    return hardware.gpus[0] if hardware.gpus else None
