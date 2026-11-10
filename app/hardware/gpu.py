"""GPU detection.

Priority order:
1. NVML via pynvml for live telemetry.
2. PyTorch CUDA properties for compute capability and SM count.
3. ``nvidia-smi`` CSV parsing when NVML is unavailable.

A detected NVIDIA GPU is reported even when the installed PyTorch build cannot
use CUDA. In that case ``cuda_available`` stays false with an explanatory note.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from app.core.config import BYTES_PER_MIB
from app.schemas import GPUInfo

logger = logging.getLogger(__name__)


def _clean_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _torch_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpus": [],
        "notes": [],
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["notes"].append(f"PyTorch could not be imported: {exc}")
        return [], info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.cuda.is_available failed: %s", exc)
        available = False
        info["notes"].append(f"CUDA availability check failed: {exc}")
    info["cuda_available"] = available
    if not available:
        return [], info

    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        info["notes"].append(f"Could not enumerate CUDA devices: {exc}")
        return [], info

    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            entry: dict[str, Any] = {
                "index": index,
                "name": _clean_name(props.name),
                "total_memory_mb": round(float(props.total_memory) / BYTES_PER_MIB, 1),
                "multiprocessors": int(getattr(props, "multi_processor_count", 0)) or None,
                "compute_capability": f"{props.major}.{props.minor}",
                "source": "PyTorch CUDA",
            }
            clock = getattr(props, "clock_rate", None)
            if clock:
                # torch reports kHz on supported builds.
                entry["clock_mhz"] = round(float(clock) / 1000.0, 1)
            info["gpus"].append(entry)
        except Exception as exc:  # noqa: BLE001
            info["notes"].append(f"Could not read CUDA properties for GPU {index}: {exc}")
    return info["gpus"], info


def _nvml_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: dict[str, Any] = {
        "driver_version": None,
        "cuda_driver_version": None,
        "gpus": [],
        "available": False,
        "notes": [],
    }
    try:
        import pynvml
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"pynvml is not available: {exc}")
        return [], result

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML initialization failed: %s", exc)
        result["notes"].append(f"NVML initialization failed: {exc}")
        return [], result

    result["available"] = True
    try:
        try:
            result["driver_version"] = _clean_name(pynvml.nvmlSystemGetDriverVersion())
        except Exception:  # noqa: BLE001
            pass
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            major, minor = divmod(int(cuda_version), 1000)
            result["cuda_driver_version"] = f"{major}.{minor}"
        except Exception:  # noqa: BLE001
            pass

        count = int(pynvml.nvmlDeviceGetCount())
        for index in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                entry: dict[str, Any] = {"index": index, "source": "NVML"}
                entry["name"] = _clean_name(pynvml.nvmlDeviceGetName(handle))
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                entry["total_memory_mb"] = round(memory.total / BYTES_PER_MIB, 1)
                entry["used_memory_mb"] = round(memory.used / BYTES_PER_MIB, 1)
                entry["free_memory_mb"] = round(memory.free / BYTES_PER_MIB, 1)
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    entry["utilization_pct"] = float(utilization.gpu)
                    entry["memory_utilization_pct"] = float(utilization.memory)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    entry["compute_capability"] = f"{major}.{minor}"
                except Exception:  # noqa: BLE001
                    pass
                try:
                    entry["temperature_c"] = float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    entry["power_w"] = round(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    entry["clock_mhz"] = float(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    )
                except Exception:  # noqa: BLE001
                    pass
                result["gpus"].append(entry)
            except Exception as exc:  # noqa: BLE001
                logger.debug("NVML read failed for GPU %s: %s", index, exc)
                result["notes"].append(f"NVML read failed for GPU {index}: {exc}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return result["gpus"], result


def _smi_snapshot() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    query = (
        "index,name,memory.total,memory.used,utilization.gpu,compute_cap,"
        "driver_version,temperature.gpu,clocks.sm"
    )
    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi fallback failed: %s", exc)
        return []

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        entry: dict[str, Any] = {
            "index": index,
            "name": parts[1],
            "total_memory_mb": _safe_float(parts[2]),
            "source": "nvidia-smi",
        }
        optional = {
            "used_memory_mb": 3,
            "utilization_pct": 4,
            "compute_capability": 5,
            "driver_version": 6,
            "temperature_c": 7,
            "clock_mhz": 8,
        }
        for key, position in optional.items():
            if position < len(parts) and parts[position].upper() not in {"N/A", "[N/A]"}:
                if key == "compute_capability":
                    entry[key] = parts[position]
                else:
                    entry[key] = _safe_float(parts[position])
        total = entry.get("total_memory_mb")
        used = entry.get("used_memory_mb")
        if total is not None and used is not None:
            entry["free_memory_mb"] = round(float(total) - float(used), 1)
        gpus.append(entry)
    return gpus


def _merge_entries(base: dict[int, dict[str, Any]], extra: list[dict[str, Any]]) -> None:
    for entry in extra:
        index = int(entry["index"])
        if index not in base:
            base[index] = dict(entry)
            continue
        for key, value in entry.items():
            if value is not None and base[index].get(key) in (None, "", 0):
                base[index][key] = value
        sources = {base[index].get("source", ""), entry.get("source", "")}
        base[index]["source"] = "+".join(sorted(source for source in sources if source))


def detect_gpus() -> tuple[list[GPUInfo], dict[str, Any]]:
    """Return detected GPUs plus runtime/driver metadata. Never raises."""
    notes: list[str] = []
    torch_gpus, torch_info = _torch_snapshot()
    nvml_gpus, nvml_info = _nvml_snapshot()
    smi_gpus = _smi_snapshot()

    merged: dict[int, dict[str, Any]] = {}
    _merge_entries(merged, smi_gpus)
    _merge_entries(merged, nvml_gpus)
    _merge_entries(merged, torch_gpus)

    for entry in merged.values():
        if entry.get("multiprocessors") is None:
            entry.setdefault("notes", [])
            entry["notes"].append(
                "SM count unavailable because PyTorch CUDA properties could not be read."
            )
        if entry.get("free_memory_mb") is None and entry.get("total_memory_mb"):
            used = entry.get("used_memory_mb") or 0.0
            entry["free_memory_mb"] = round(
                max(0.0, float(entry["total_memory_mb"]) - float(used)), 1
            )

    gpus: list[GPUInfo] = []
    for index in sorted(merged):
        entry = merged[index]
        entry_notes = list(dict.fromkeys(entry.get("notes", [])))
        if not torch_info["cuda_available"]:
            entry_notes.append(
                "The NVIDIA driver or GPU is visible, but this PyTorch installation "
                "cannot use CUDA (CPU build, missing driver, or unsupported device)."
            )
        try:
            gpus.append(
                GPUInfo(
                    index=index,
                    name=_clean_name(entry.get("name", "Unknown GPU")),
                    total_memory_mb=float(entry.get("total_memory_mb") or 0.0),
                    used_memory_mb=_safe_float(entry.get("used_memory_mb")),
                    free_memory_mb=_safe_float(entry.get("free_memory_mb")),
                    utilization_pct=_safe_float(entry.get("utilization_pct")),
                    memory_utilization_pct=_safe_float(entry.get("memory_utilization_pct")),
                    compute_capability=entry.get("compute_capability"),
                    multiprocessors=entry.get("multiprocessors"),
                    clock_mhz=_safe_float(entry.get("clock_mhz")),
                    temperature_c=_safe_float(entry.get("temperature_c")),
                    power_w=_safe_float(entry.get("power_w")),
                    driver_version=(
                        entry.get("driver_version") or nvml_info.get("driver_version")
                    ),
                    cuda_available=bool(torch_info["cuda_available"]),
                    source=entry.get("source", "unknown"),
                    notes=entry_notes,
                )
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Could not normalize GPU {index} data: {exc}")

    if not gpus:
        notes.append(
            "No NVIDIA GPU was detected. CPU-only estimates and benchmarks remain "
            "fully available; GPU requirements are still estimated hypothetically."
        )
    if nvml_info.get("notes"):
        notes.extend(nvml_info["notes"])
    if torch_info.get("notes"):
        notes.extend(torch_info["notes"])
    notes = list(dict.fromkeys(notes))

    metadata = {
        "torch_available": bool(torch_info["torch_available"]),
        "torch_version": torch_info.get("torch_version"),
        "torch_cuda_version": torch_info.get("torch_cuda_version"),
        "cuda_available": bool(torch_info["cuda_available"]),
        "nvidia_driver_version": (
            nvml_info.get("driver_version")
            or next((gpu.driver_version for gpu in gpus if gpu.driver_version), None)
        ),
        "notes": notes,
    }
    return gpus, metadata
