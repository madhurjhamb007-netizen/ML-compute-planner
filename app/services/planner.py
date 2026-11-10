"""Planner service: combine detection, estimation, and recommendations."""

from __future__ import annotations

import time

from app.core.config import DISCLAIMER
from app.estimation.flops import estimate_compute
from app.estimation.memory import estimate_memory
from app.estimation.recommender import (
    build_bottlenecks,
    build_recommendations,
    build_suitability,
)
from app.estimation.runtime import resolve_precision
from app.estimation.time_predictor import predict_training_time
from app.hardware.detector import detect_hardware
from app.schemas import (
    ComputeEstimate,
    EstimateResponse,
    HardwareInfo,
    MemoryEstimate,
    ModelConfig,
    RecommendationSet,
    TimeEstimate,
)

_ASSUMPTIONS = [
    "FLOPs count one multiply-add as two operations; elementwise activations, "
    "normalization, pooling, and optimizer math are excluded unless noted.",
    "Training FLOPs are approximated as 3x forward FLOPs (forward + backward).",
    "Memory includes estimated weights, gradients, optimizer state, activations for "
    "one batch, input buffers, dataset cache, and runtime overhead.",
    "CUDA context overhead uses a generic 512 MiB allowance; CPU runtime overhead uses 256 MiB.",
    "CPU/GPU peak throughput is derived from detected cores/frequency or SM count/clock "
    "with generic lane assumptions; it is not a device specification lookup.",
    "Achieved efficiency and time predictions are heuristics with intentionally wide "
    "uncertainty ranges. Use the benchmark to calibrate on this host.",
    "Dataset memory assumes the full dataset is materialized in host RAM.",
    "Hardware facts are detected at request time. Unavailable values stay null or are "
    "explicitly labelled as estimates.",
]


def estimate_plan(
    config: ModelConfig, hardware: HardwareInfo | None = None
) -> EstimateResponse:
    started = time.perf_counter()
    hardware = hardware or detect_hardware()
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, hardware)
    time_estimate = predict_training_time(
        config, compute, hardware, memory.target_device
    )
    suitability = build_suitability(config, compute, memory, time_estimate, hardware)
    bottlenecks = build_bottlenecks(config, compute, memory, time_estimate, hardware)
    recommendations = build_recommendations(
        config, compute, memory, time_estimate, hardware
    )

    warnings: list[str] = list(hardware.warnings)
    resolution = resolve_precision(config, hardware, memory.target_device)
    if resolution.effective != config.precision:
        warnings.append(
            f"Requested precision {config.precision} is not used; effective precision is "
            f"{resolution.effective}."
        )
    if memory.target_device == "cpu" and config.device_preference == "cuda":
        warnings.append(
            "CUDA was requested but the workload was assigned to CPU. Check GPU "
            "availability, CUDA support in PyTorch, and memory fit."
        )
    if not memory.fits_target:
        warnings.append(
            "The estimated workload does not fit in the target device's usable memory. "
            "Review the memory section before running."
        )
    if config.model_type == "random_forest":
        warnings.append(
            "Random Forest estimates ignore epochs and precision; it is a CPU tree workload."
        )
    warnings = list(dict.fromkeys(warnings))

    return EstimateResponse(
        config=config,
        hardware=hardware,
        compute=compute,
        memory=memory,
        time=time_estimate,
        recommendations=recommendations,
        suitability=suitability,
        bottlenecks=bottlenecks,
        assumptions=_ASSUMPTIONS,
        warnings=warnings,
        disclaimer=DISCLAIMER,
        estimation_seconds=round(time.perf_counter() - started, 3),
    )


def get_compute(config: ModelConfig) -> ComputeEstimate:
    return estimate_compute(config)


def get_memory(config: ModelConfig) -> MemoryEstimate:
    hardware = detect_hardware()
    compute = estimate_compute(config)
    return estimate_memory(config, compute, hardware)


def get_time(config: ModelConfig) -> TimeEstimate:
    hardware = detect_hardware()
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, hardware)
    return predict_training_time(config, compute, hardware, memory.target_device)


def get_recommendations(config: ModelConfig) -> RecommendationSet:
    return estimate_plan(config).recommendations
