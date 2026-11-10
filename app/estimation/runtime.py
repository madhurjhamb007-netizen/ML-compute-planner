"""Execution-context resolution: precision support and honest fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.hardware.detector import primary_gpu
from app.schemas import HardwareInfo, ModelConfig, Precision, ResolvedDevice


@dataclass
class PrecisionResolution:
    requested: Precision
    effective: Precision
    notes: list[str] = field(default_factory=list)


def _capability_tuple(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def resolve_precision(
    config: ModelConfig, hardware: HardwareInfo, device: ResolvedDevice
) -> PrecisionResolution:
    """Resolve whether the requested precision is usable.

    PyTorch autocast keeps master weights in FP32. On CPU, FP16 is not supported
    by this application; BF16 is allowed experimentally but no speedup is promised.
    On CUDA, FP16 requires compute capability >= 5.3 and BF16 requires >= 8.0.
    """
    requested = config.precision
    notes: list[str] = []
    if config.model_type == "random_forest":
        return PrecisionResolution(
            requested=requested,
            effective="fp32",
            notes=["Random Forest is a CPU tree model; precision settings do not apply."],
        )

    if device == "cpu":
        if requested == "fp16":
            notes.append("FP16 is not supported for CPU training; FP32 is used instead.")
            return PrecisionResolution(requested, "fp32", notes)
        if requested == "bf16":
            notes.append(
                "CPU BF16 autocast is experimental and may be slower than FP32 on "
                "this processor. No speedup is assumed in predictions."
            )
        return PrecisionResolution(requested, requested, notes)

    gpu = primary_gpu(hardware)
    if gpu is None:
        notes.append("No GPU is available; requested mixed precision falls back to FP32.")
        return PrecisionResolution(requested, "fp32", notes)
    capability = _capability_tuple(gpu.compute_capability)
    if requested == "fp16":
        if capability is not None and capability < (5, 3):
            notes.append(
                f"GPU compute capability {gpu.compute_capability} does not support "
                "efficient FP16; FP32 is used."
            )
            return PrecisionResolution(requested, "fp32", notes)
        if capability is None:
            notes.append(
                "GPU compute capability is unknown; FP16 is attempted with autocast."
            )
        return PrecisionResolution(requested, "fp16", notes)
    if requested == "bf16":
        if capability is not None and capability < (8, 0):
            notes.append(
                f"GPU compute capability {gpu.compute_capability} does not support "
                "BF16; FP32 is used."
            )
            return PrecisionResolution(requested, "fp32", notes)
        if capability is None:
            notes.append(
                "GPU compute capability is unknown; BF16 is attempted with autocast."
            )
        return PrecisionResolution(requested, "bf16", notes)
    return PrecisionResolution(requested, "fp32", notes)
