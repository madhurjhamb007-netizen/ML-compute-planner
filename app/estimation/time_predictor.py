"""Heuristic training-time prediction.

This is a planning model, not a simulator. It combines:
* a theoretical peak throughput derived from detected CPU frequency/core count or
  GPU SM count/clock,
* family-specific achieved-efficiency assumptions,
* batch-size and model-size penalties,
* per-optimizer-step overhead and a startup allowance.

Predictions are deliberately returned as a wide range. The benchmark endpoint
exists precisely because these assumptions must be checked against reality.
"""

from __future__ import annotations

import math

from app.core.config import (
    DEFAULT_GPU_CLOCK_MHZ,
    GPU_FP32_LANES_PER_SM,
    MEMORY_HEADROOM_FRACTION,
)
from app.estimation.flops import _rf_effective_depth, estimate_compute
from app.estimation.memory import estimate_memory
from app.estimation.runtime import resolve_precision
from app.hardware.detector import primary_gpu
from app.schemas import (
    ComputeEstimate,
    HardwareInfo,
    ModelConfig,
    Precision,
    ResolvedDevice,
    TimeEstimate,
)

_FAMILY_EFFICIENCY_CPU: dict[str, float] = {
    "linear_regression": 0.50,
    "mlp": 0.32,
    "cnn": 0.28,
    "lstm": 0.16,
    "transformer": 0.22,
}
_FAMILY_EFFICIENCY_GPU: dict[str, float] = {
    "linear_regression": 0.18,
    "mlp": 0.26,
    "cnn": 0.35,
    "lstm": 0.20,
    "transformer": 0.30,
}


def _batch_factor(batch_size: int, target_device: str) -> float:
    reference = 128.0 if target_device == "cuda" else 64.0
    return max(0.30, min(1.0, 0.40 + 0.60 * batch_size / reference))


def _size_factor(params: int) -> float:
    # Very small models underutilize hardware; large models approach nominal efficiency.
    return max(0.45, min(1.0, 0.55 + 0.15 * math.log10(max(params, 1))))


def _cpu_peak_gflops(hardware: HardwareInfo) -> tuple[float, int]:
    cpu = hardware.cpu
    threads = max(1, cpu.recommended_threads)
    thread_fraction = min(1.0, threads / max(1, cpu.physical_cores)) ** 0.9
    return cpu.estimated_fp32_gflops * thread_fraction, threads


def _gpu_peak_tflops(hardware: HardwareInfo) -> tuple[float, str | None]:
    gpu = primary_gpu(hardware)
    if gpu is None or not hardware.cuda_available:
        return 0.0, "CUDA is unavailable."
    if not gpu.multiprocessors:
        return 2.0, (
            "SM count is unavailable; a conservative generic 2 TFLOP/s FP32 peak is "
            "assumed for planning."
        )
    clock_mhz = gpu.clock_mhz or DEFAULT_GPU_CLOCK_MHZ
    peak = (
        gpu.multiprocessors
        * GPU_FP32_LANES_PER_SM
        * 2.0
        * clock_mhz
        * 1_000_000.0
    ) / 1e12
    note = (
        f"GPU peak assumes {GPU_FP32_LANES_PER_SM} FP32 lanes per SM "
        "(a conservative generic architecture assumption), not a device lookup."
    )
    return peak, note


def _rf_estimate(config: ModelConfig, hardware: HardwareInfo) -> tuple[float, float]:
    """Return (compute_seconds, overhead_seconds) for Random Forest."""
    depth = _rf_effective_depth(config)
    features_tried = max(1, round(math.sqrt(config.features)))
    work = config.n_estimators * config.dataset_size * depth * features_tried
    cores = max(1, hardware.cpu.recommended_threads)
    rate_per_core = 1.8e7  # coarse work units/second/core for a Python/NumPy forest
    compute_seconds = work / (cores * rate_per_core)
    overhead_seconds = 0.10 + config.n_estimators * 0.003
    return compute_seconds, overhead_seconds


def predict_training_time(
    config: ModelConfig,
    compute: ComputeEstimate,
    hardware: HardwareInfo,
    target_device: ResolvedDevice,
    precision_override: Precision | None = None,
    include_startup: bool = True,
) -> TimeEstimate:
    notes: list[str] = []
    target = target_device
    resolution = resolve_precision(config, hardware, target)
    precision = precision_override or resolution.effective
    notes.extend(resolution.notes)

    if config.model_type == "random_forest":
        target = "cpu"
        precision = "fp32"
        compute_seconds, overhead_seconds = _rf_estimate(config, hardware)
        steps = 0
        effective_tflops = 0.0
        confidence = "low"
        bottleneck = "CPU tree construction and memory bandwidth"
        notes.extend(
            [
                "Random Forest time is predicted with a coarse work-per-core model.",
                "Epochs are ignored; the estimate covers one forest fit.",
                "Confidence is low because tree implementations and split strategies vary widely.",
            ]
        )
        predicted = compute_seconds + overhead_seconds
        predicted_low = predicted * 0.35
        predicted_high = predicted * 2.5
        samples = config.dataset_size
    elif target == "cpu":
        peak_gflops, threads = _cpu_peak_gflops(hardware)
        efficiency = _FAMILY_EFFICIENCY_CPU.get(config.model_type, 0.25)
        effective_flops = (
            peak_gflops
            * 1e9
            * efficiency
            * _batch_factor(config.batch_size, "cpu")
            * _size_factor(compute.parameter_count)
        )
        compute_seconds = compute.total_training_flops / max(effective_flops, 1.0)
        steps = compute.optimizer_steps
        startup = 0.4 if include_startup else 0.0
        per_step = 0.0025
        overhead_seconds = startup + steps * per_step
        effective_tflops = effective_flops / 1e12
        confidence = "low" if config.model_type == "lstm" else "medium"
        bottleneck = "CPU compute throughput"
        if config.model_type in ("cnn", "transformer", "lstm"):
            bottleneck = "CPU compute for deep model (GPU would likely be faster)"
        notes.extend(
            [
                f"CPU planning peak is {peak_gflops:.1f} GFLOP/s using {threads} recommended threads.",
                "The peak figure is derived from core count and reported frequency and is "
                "not a measured benchmark.",
                "Per-step framework overhead is approximated at 2.5 ms per optimizer step.",
            ]
        )
        if config.batch_size < 16:
            bottleneck = "Small batch / CPU underutilization"
            notes.append("Very small batches typically underutilize vectorized CPU kernels.")
        predicted = compute_seconds + overhead_seconds
        predicted_low = predicted * 0.55
        predicted_high = predicted * 1.9
        samples = config.dataset_size * config.epochs
    else:
        peak_tflops, peak_note = _gpu_peak_tflops(hardware)
        if peak_note:
            notes.append(peak_note)
        efficiency = _FAMILY_EFFICIENCY_GPU.get(config.model_type, 0.25)
        precision_factor = 1.25 if precision in ("fp16", "bf16") else 1.0
        effective_tflops = (
            peak_tflops
            * efficiency
            * _batch_factor(config.batch_size, "cuda")
            * _size_factor(compute.parameter_count)
            * precision_factor
        )
        effective_flops = effective_tflops * 1e12
        compute_seconds = compute.total_training_flops / max(effective_flops, 1.0)
        steps = compute.optimizer_steps
        overhead_seconds = (1.0 if include_startup else 0.0) + steps * 0.004
        bottleneck = "GPU compute/launch utilization"
        if peak_tflops and effective_tflops > 0 and effective_tflops / peak_tflops < 0.15:
            bottleneck = "Small workload / GPU launch overhead"
        confidence = "medium" if peak_tflops > 0 else "low"
        notes.extend(
            [
                f"GPU planning peak is {peak_tflops:.2f} TFLOP/s; effective planning "
                f"throughput is {effective_tflops:.2f} TFLOP/s.",
                "A conservative precision speedup factor of 1.25× is applied for FP16/BF16; "
                "tensor-core performance can be higher or lower.",
                "GPU launch overhead is approximated at 4 ms per optimizer step plus 1 s startup.",
            ]
        )
        if config.batch_size < 32:
            bottleneck = "Small batch / GPU underutilization"
            notes.append("Small batches underutilize GPU SMs and increase launch overhead.")
        predicted = compute_seconds + overhead_seconds
        predicted_low = predicted * 0.50
        predicted_high = predicted * 2.0
        samples = config.dataset_size * config.epochs

    predicted = max(predicted, 0.001)
    throughput = samples / predicted
    return TimeEstimate(
        target_device=target,  # type: ignore[arg-type]
        precision_used=precision,
        predicted_seconds=round(predicted, 4),
        predicted_low_seconds=round(max(0.001, predicted_low), 4),
        predicted_high_seconds=round(predicted_high, 4),
        predicted_hours=round(predicted / 3600.0, 4),
        throughput_samples_per_second=round(throughput, 3),
        effective_tflops=round(effective_tflops, 4),
        compute_seconds=round(compute_seconds, 4),
        overhead_seconds=round(overhead_seconds, 4),
        confidence=confidence,  # type: ignore[arg-type]
        bottleneck_primary=bottleneck,
        notes=notes,
    )


def predict_benchmark_duration(
    effective_config: ModelConfig,
    hardware: HardwareInfo,
    target_device: ResolvedDevice,
    precision: Precision,
) -> tuple[float, float, float, TimeEstimate, ComputeEstimate]:
    """Predict a benchmark-shaped workload (one pass over the synthetic data).

    Returns (predicted, low, high, time_estimate, compute_estimate).
    """
    compute = estimate_compute(effective_config)
    time_estimate = predict_training_time(
        effective_config,
        compute,
        hardware,
        target_device,
        precision_override=precision,
        include_startup=False,
    )
    return (
        time_estimate.predicted_seconds,
        time_estimate.predicted_low_seconds,
        time_estimate.predicted_high_seconds,
        time_estimate,
        compute,
    )


def memory_fit_for_device(
    config: ModelConfig, compute: ComputeEstimate, hardware: HardwareInfo, device: str
) -> bool:
    """Small shared helper for benchmark safety checks."""
    memory = estimate_memory(config, compute, hardware)
    if device == "cuda":
        return bool(memory.fits_gpu) and memory.gpu_host_ram_mb <= (
            hardware.ram.available_mb * MEMORY_HEADROOM_FRACTION
        )
    return memory.fits_cpu
