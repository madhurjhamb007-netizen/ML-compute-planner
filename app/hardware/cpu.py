"""CPU and system memory detection with cross-platform fallbacks."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from typing import Any

import psutil

from app.core.config import BYTES_PER_GIB, BYTES_PER_MIB, DEFAULT_CPU_GHZ
from app.schemas import CPUInfo, RAMInfo

logger = logging.getLogger(__name__)


def _windows_cpu_name() -> str | None:
    try:
        import winreg

        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip() or None
    except (ImportError, OSError):
        return None


def _linux_cpu_name() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        return None
    return None


def _mac_cpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_cpu_name() -> tuple[str, list[str]]:
    notes: list[str] = []
    if sys.platform == "win32":
        name = _windows_cpu_name()
    elif sys.platform == "darwin":
        name = _mac_cpu_name()
    else:
        name = _linux_cpu_name()

    if not name:
        name = platform.processor() or platform.machine() or "Unknown CPU"
        notes.append("CPU model name could not be read; using the platform identifier.")
    return " ".join(name.split()), notes


def _cpu_frequencies() -> tuple[float | None, float | None, list[str]]:
    """Return (current MHz, max MHz, notes). Values may be unavailable in VMs."""
    notes: list[str] = []
    try:
        freq = psutil.cpu_freq()
    except (AttributeError, OSError) as exc:  # pragma: no cover - platform dependent
        logger.debug("psutil.cpu_freq failed: %s", exc)
        return None, None, ["CPU frequency is unavailable on this platform."]

    if freq is None:
        return None, None, ["CPU frequency is unavailable on this platform."]

    current = float(freq.current) if freq.current and freq.current > 0 else None
    peak = float(freq.max) if freq.max and freq.max > 0 else None
    if current is None:
        notes.append("Current CPU frequency is unavailable.")
    if peak is None:
        notes.append("Maximum CPU frequency is unavailable; a generic estimate is used.")
    if peak is not None and current is not None and peak == current:
        notes.append(
            "Reported current and maximum frequency are identical; this is common in "
            "virtual machines and may hide turbo frequency."
        )
    return current, peak, notes


def _thread_recommendation(physical_cores: int) -> int:
    """Leave a core for the OS on larger machines, keep all cores on tiny ones."""
    if physical_cores >= 4:
        return max(1, physical_cores - 1)
    return max(1, physical_cores)


def detect_cpu() -> CPUInfo:
    name, notes = detect_cpu_name()
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    if psutil.cpu_count(logical=False) is None:
        notes.append("Physical core count is unavailable; assumed from logical count.")

    try:
        per_core = [float(value) for value in psutil.cpu_percent(interval=0.1, percpu=True)]
    except (OSError, NotImplementedError) as exc:
        logger.debug("per-core utilization failed: %s", exc)
        per_core = []
        notes.append("Per-core CPU utilization is unavailable.")
    utilization = round(sum(per_core) / len(per_core), 2) if per_core else 0.0

    current_mhz, max_mhz, freq_notes = _cpu_frequencies()
    notes.extend(freq_notes)

    reference_mhz = max_mhz or current_mhz
    if reference_mhz:
        reference_ghz = reference_mhz / 1000.0
    else:
        reference_ghz = DEFAULT_CPU_GHZ
        notes.append(
            f"CPU frequency unknown; peak throughput assumes {DEFAULT_CPU_GHZ:.1f} GHz."
        )

    # Generic AVX2-class FP32 estimate: 8 lanes * FMA (2 FLOPs) = 16 FLOPs/cycle/core.
    # This is a theoretical upper bound, not a measured value.
    estimated_gflops = round(physical * reference_ghz * 16.0, 1)
    notes.append(
        "Theoretical peak assumes 16 FP32 FLOPs/core/cycle (AVX2 FMA); actual "
        "throughput is usually much lower."
    )

    return CPUInfo(
        name=name,
        architecture=platform.machine() or "unknown",
        physical_cores=physical,
        logical_threads=logical,
        current_frequency_mhz=current_mhz,
        max_frequency_mhz=max_mhz,
        utilization_pct=utilization,
        per_core_utilization_pct=[round(value, 2) for value in per_core],
        recommended_threads=_thread_recommendation(physical),
        estimated_fp32_gflops=estimated_gflops,
        notes=notes,
    )


def detect_ram() -> RAMInfo:
    memory = psutil.virtual_memory()
    return RAMInfo(
        total_mb=round(memory.total / BYTES_PER_MIB, 1),
        available_mb=round(memory.available / BYTES_PER_MIB, 1),
        used_mb=round(memory.used / BYTES_PER_MIB, 1),
        utilization_pct=round(float(memory.percent), 2),
    )


def process_memory_mb() -> float:
    """Current process RSS in MiB (used by the benchmark sampler)."""
    return round(psutil.Process().memory_info().rss / BYTES_PER_MIB, 1)


def system_ram_total_gib() -> float:
    return round(psutil.virtual_memory().total / BYTES_PER_GIB, 2)


def safe_call(function: Any, *args: Any, default: Any = None) -> Any:
    """Call a detection helper without letting one failure break the snapshot."""
    try:
        return function(*args)
    except Exception as exc:  # noqa: BLE001 - hardware backends are unreliable
        logger.debug("Hardware detection call %s failed: %s", function, exc)
        return default
