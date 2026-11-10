"""Benchmark orchestration, safety enforcement, and prediction comparison."""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.benchmark.monitor import ResourceSampler
from app.benchmark.workloads import (
    RandomForestWorkload,
    TorchWorkload,
    WorkloadStats,
)
from app.core.config import (
    BENCHMARK_DEFAULT_SAMPLES,
    BENCHMARK_MAX_BATCH_SIZE,
    BENCHMARK_MAX_RF_DEPTH,
    BENCHMARK_MAX_RF_SAMPLES,
    BENCHMARK_MAX_RF_TREES,
    BENCHMARK_MAX_SAMPLES,
    BENCHMARK_MAX_SECONDS,
    BENCHMARK_MAX_STEPS,
    BENCHMARK_MAX_WARMUP_STEPS,
    DISCLAIMER,
)
from app.estimation.flops import estimate_compute
from app.estimation.memory import estimate_memory
from app.estimation.runtime import resolve_precision
from app.estimation.time_predictor import (
    predict_benchmark_duration,
    predict_training_time,
)
from app.hardware.cpu import process_memory_mb
from app.hardware.detector import detect_hardware
from app.schemas import (
    BenchmarkComparison,
    BenchmarkOptions,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    BenchmarkSample,
    HardwareInfo,
    ModelConfig,
    Precision,
    ResolvedDevice,
)

logger = logging.getLogger(__name__)


class BenchmarkSafetyError(RuntimeError):
    """The requested benchmark is unsafe or cannot fit in available resources."""


class BenchmarkExecutionError(RuntimeError):
    """The benchmark started but failed while executing."""


@dataclass
class BenchmarkPlan:
    mode: Literal["torch", "rf"]
    device: ResolvedDevice
    precision: Precision
    effective_config: ModelConfig
    samples: int
    batch_size: int
    steps: int
    warmup_steps: int
    warnings: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)


def _rf_plan(
    request: BenchmarkRequest, hardware: HardwareInfo
) -> BenchmarkPlan:
    warnings: list[str] = []
    config = ModelConfig(**request.model_dump(exclude={"benchmark"}))
    options: BenchmarkOptions = request.benchmark
    if options.device == "cuda":
        warnings.append("Random Forest runs on the CPU; the CUDA request was ignored.")

    samples = min(
        options.samples or BENCHMARK_DEFAULT_SAMPLES,
        BENCHMARK_MAX_RF_SAMPLES,
        config.dataset_size,
    )
    samples = max(1, samples)
    features = min(config.features, 256)
    n_estimators = max(1, min(config.n_estimators, BENCHMARK_MAX_RF_TREES))
    max_depth = max(1, min(config.max_depth, BENCHMARK_MAX_RF_DEPTH))
    if features != config.features:
        warnings.append(f"Features were capped from {config.features} to {features} for safety.")
    if n_estimators != config.n_estimators:
        warnings.append(
            f"Trees were capped from {config.n_estimators} to {n_estimators} for safety."
        )
    if max_depth != config.max_depth:
        warnings.append(f"max_depth was capped from {config.max_depth} to {max_depth}.")

    effective = config.model_copy(
        update={
            "dataset_size": samples,
            "epochs": 1,
            "features": features,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "precision": "fp32",
            "device_preference": "cpu",
        }
    )
    compute = estimate_compute(effective)
    memory = estimate_memory(effective, compute, hardware)
    if not memory.fits_cpu:
        raise BenchmarkSafetyError(
            "The Random Forest benchmark is estimated to exceed available RAM. "
            "Reduce samples, features, trees, or max_depth."
        )

    predicted = predict_training_time(
        effective, compute, hardware, "cpu", precision_override="fp32"
    ).predicted_seconds
    if predicted > options.max_seconds:
        scaled = max(1, int(n_estimators * (options.max_seconds * 0.8) / predicted))
        if scaled < n_estimators:
            n_estimators = scaled
            warnings.append(
                f"Trees reduced to {n_estimators} to stay within the benchmark time budget."
            )
            effective = effective.model_copy(update={"n_estimators": n_estimators})
            compute = estimate_compute(effective)
            memory = estimate_memory(effective, compute, hardware)

    safety_notes = [
        f"Random Forest benchmark uses {samples} synthetic samples, {features} features, "
        f"{n_estimators} trees, and max_depth {max_depth}.",
        "Only one CPU training pass is timed; there is no warmup for tree fitting.",
        "Tree fitting uses scikit-learn when installed, with a bundled NumPy forest "
        "fallback so the platform remains fully functional without it.",
    ]
    return BenchmarkPlan(
        mode="rf",
        device="cpu",
        precision="fp32",
        effective_config=effective,
        samples=samples,
        batch_size=max(1, min(effective.batch_size, BENCHMARK_MAX_BATCH_SIZE)),
        steps=1,
        warmup_steps=0,
        warnings=warnings,
        safety_notes=safety_notes,
    )


def _torch_plan(
    config: ModelConfig, options: BenchmarkOptions, hardware: HardwareInfo
) -> BenchmarkPlan:
    warnings: list[str] = []
    requested = options.device
    device: ResolvedDevice
    if requested == "cpu":
        device = "cpu"
    elif requested == "cuda" and not hardware.cuda_available:
        device = "cpu"
        warnings.append("CUDA was requested but is unavailable; the benchmark falls back to CPU.")
    else:
        device = "cuda" if hardware.cuda_available else "cpu"

    max_seconds = min(options.max_seconds, BENCHMARK_MAX_SECONDS)
    sample_limit = max(
        1,
        min(
            options.samples or BENCHMARK_DEFAULT_SAMPLES,
            BENCHMARK_MAX_SAMPLES,
            config.dataset_size,
        ),
    )
    batch_size = max(
        1,
        min(options.batch_size or config.batch_size, BENCHMARK_MAX_BATCH_SIZE, sample_limit),
    )
    available_full_steps = max(1, sample_limit // batch_size)
    initial_steps = max(
        1, min(options.steps, BENCHMARK_MAX_STEPS, available_full_steps)
    )
    warmup_steps = min(options.warmup_steps, BENCHMARK_MAX_WARMUP_STEPS)

    # Outer loop permits exactly one auto fallback CUDA -> CPU.
    for _attempt in range(2):
        precision_res = resolve_precision(config, hardware, device)
        precision = precision_res.effective
        warnings.extend(precision_res.notes)
        steps = initial_steps
        rebuild_device = False

        while True:
            samples_seen = steps * batch_size
            effective = config.model_copy(
                update={
                    "dataset_size": samples_seen,
                    "epochs": 1,
                    "batch_size": batch_size,
                    "precision": precision,
                    "device_preference": device,
                }
            )
            compute = estimate_compute(effective)
            memory = estimate_memory(effective, compute, hardware)
            if device == "cuda":
                fits = bool(memory.fits_gpu) and memory.gpu_host_ram_mb <= (
                    hardware.ram.available_mb * 0.90
                )
            else:
                fits = memory.fits_cpu
            if not fits:
                message = (
                    "The benchmark workload is estimated not to fit in "
                    f"{'GPU VRAM + host RAM' if device == 'cuda' else 'available RAM'}. "
                    "Reduce model size, batch size, or samples."
                )
                if device == "cuda" and requested == "auto":
                    device = "cpu"
                    warnings.append(
                        "Auto device selection fell back to CPU because the benchmark "
                        "did not fit in GPU memory."
                    )
                    rebuild_device = True
                    break
                raise BenchmarkSafetyError(message)

            time_estimate = predict_training_time(
                effective, compute, hardware, device, precision_override=precision
            )
            if time_estimate.predicted_seconds > max_seconds and steps > 1:
                scaled = (max_seconds * 0.8) / time_estimate.predicted_seconds
                new_steps = max(1, int(steps * scaled))
                if new_steps < steps:
                    warnings.append(
                        f"Benchmark steps reduced from {steps} to {new_steps} to respect "
                        "the time budget."
                    )
                    steps = new_steps
                    continue
            if steps == 1 and time_estimate.predicted_seconds > max_seconds * 2.0:
                raise BenchmarkSafetyError(
                    "A single predicted benchmark step exceeds the safety time budget. "
                    "Reduce batch size, sequence length, image size, or model dimensions."
                )
            warnings = list(dict.fromkeys(warnings))
            safety_notes = [
                f"Benchmark ran {steps} measured step(s) + {warmup_steps} warmup step(s) "
                f"at batch size {batch_size} on {device.upper()}.",
                "One synthetic batch is reused; data loading, disk I/O, and preprocessing "
                "are excluded from the measurement.",
                "CPU utilization is system-wide and can include unrelated processes. "
                "Process CPU is also reported.",
                "CUDA timings are synchronized before and after the measured loop.",
            ]
            return BenchmarkPlan(
                mode="torch",
                device=device,
                precision=precision,
                effective_config=effective,
                samples=samples_seen,
                batch_size=batch_size,
                steps=steps,
                warmup_steps=warmup_steps,
                warnings=warnings,
                safety_notes=safety_notes,
            )
        if rebuild_device:
            continue
    raise BenchmarkSafetyError("Could not construct a safe benchmark plan.")


def build_plan(
    request: BenchmarkRequest, hardware: HardwareInfo
) -> BenchmarkPlan:
    config = ModelConfig(**request.model_dump(exclude={"benchmark"}))
    if config.model_type == "random_forest":
        return _rf_plan(request, hardware)
    return _torch_plan(config, request.benchmark, hardware)


def _comparison(
    effective_config: ModelConfig,
    hardware: HardwareInfo,
    device: ResolvedDevice,
    precision: Precision,
    measured_seconds: float,
    measured_samples_per_second: float,
    measured_tflops: float,
    compute_flops: int,
) -> BenchmarkComparison:
    predicted, low, high, _, _ = predict_benchmark_duration(
        effective_config, hardware, device, precision
    )
    predicted = max(predicted, 1e-6)
    measured = max(measured_seconds, 1e-6)
    ratio = measured / predicted
    error = (measured - predicted) / predicted * 100.0
    predicted_tflops = compute_flops / predicted / 1e12
    within = low <= measured <= high
    if within:
        interpretation = (
            "Measured duration is inside the prediction range. The heuristic model is "
            "calibrated reasonably for this short synthetic workload."
        )
    elif measured > predicted:
        interpretation = (
            f"Measured is {error:.0f}% slower than the point prediction. Framework overhead, "
            "thread scheduling, or low arithmetic intensity is likely under-predicted."
        )
    else:
        interpretation = (
            f"Measured is {abs(error):.0f}% faster than the point prediction. The heuristic "
            "efficiency assumption was conservative for this workload."
        )
    if effective_config.model_type == "random_forest":
        interpretation += (
            " Random Forest predictions have low confidence; this comparison is most "
            "useful as a calibration point."
        )
    return BenchmarkComparison(
        predicted_seconds=round(predicted, 4),
        predicted_low_seconds=round(low, 4),
        predicted_high_seconds=round(high, 4),
        measured_seconds=round(measured, 6),
        time_ratio=round(ratio, 4),
        time_error_pct=round(error, 2),
        predicted_samples_per_second=round(effective_config.dataset_size / predicted, 3),
        measured_samples_per_second=round(measured_samples_per_second, 3),
        throughput_ratio=round(
            measured_samples_per_second / max(effective_config.dataset_size / predicted, 1e-9),
            4,
        ),
        predicted_tflops=round(predicted_tflops, 5),
        measured_tflops=round(measured_tflops, 5),
        within_expected_range=within,
        interpretation=interpretation,
    )


def run_benchmark(
    request: BenchmarkRequest, hardware: HardwareInfo | None = None
) -> BenchmarkResponse:
    hardware = hardware or detect_hardware()
    plan = build_plan(request, hardware)
    started_at = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:12]
    baseline_ram = process_memory_mb()
    gpu_index = (
        hardware.gpus[0].index
        if plan.device == "cuda" and hardware.gpus
        else None
    )
    sampler = ResourceSampler(gpu_index)
    workload = None
    stats: WorkloadStats | None = None
    summary = None
    warnings = list(plan.warnings)
    safety_notes = list(plan.safety_notes)

    logger.info(
        "Starting benchmark %s: model=%s device=%s precision=%s samples=%s steps=%s",
        run_id,
        plan.effective_config.model_type,
        plan.device,
        plan.precision,
        plan.samples,
        plan.steps,
    )
    started_clock = time.perf_counter()
    try:
        if plan.mode == "rf":
            workload = RandomForestWorkload(
                plan.effective_config,
                samples=plan.samples,
                max_seconds=request.benchmark.max_seconds,
                n_jobs=hardware.cpu.recommended_threads,
            )
            workload.prepare()
            prepared_ram = process_memory_mb()
            sampler.start()
            stats = workload.measure()
        else:
            workload = TorchWorkload(
                plan.effective_config,
                plan.device,
                plan.precision,
                plan.batch_size,
                plan.steps,
                plan.warmup_steps,
                request.benchmark.max_seconds,
            )
            workload.prepare()
            prepared_ram = process_memory_mb()
            sampler.start()
            stats = workload.measure()
        summary = sampler.stop()
    except BenchmarkSafetyError:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            summary = sampler.stop()
        except Exception:  # noqa: BLE001
            summary = None
        logger.exception("Benchmark %s failed during execution", run_id)
        raise BenchmarkExecutionError(
            f"Benchmark execution failed: {exc}. Check available memory and reduce the "
            "workload size."
        ) from exc
    finally:
        if workload is not None:
            try:
                workload.cleanup()
            except Exception:  # noqa: BLE001
                logger.debug("Benchmark cleanup failed", exc_info=True)

    total_wall = round(time.perf_counter() - started_clock, 6)
    if stats is None or summary is None:
        raise BenchmarkExecutionError("Benchmark produced no measurements.")
    duration = stats.duration_seconds
    if duration <= 0:
        raise BenchmarkExecutionError("Benchmark completed too quickly to measure reliably.")

    effective_config = plan.effective_config
    if stats.samples_seen != effective_config.dataset_size:
        warnings.append(
            f"The measured run processed {stats.samples_seen} samples instead of the "
            f"planned {effective_config.dataset_size}; comparison uses the measured count."
        )
        effective_config = effective_config.model_copy(
            update={"dataset_size": max(1, stats.samples_seen)}
        )
    compute = estimate_compute(effective_config)
    samples_per_second = stats.samples_seen / duration
    measured_tflops = compute.total_training_flops / duration / 1e12
    avg_step_ms = duration / max(1, stats.steps_completed) * 1000.0
    ram_peak = max(prepared_ram, summary.ram_peak_mb)
    ram_delta = max(0.0, ram_peak - baseline_ram)
    derived_process_cpu = (
        stats.cpu_time_seconds / duration * 100.0 if stats.cpu_time_seconds else 0.0
    )
    process_cpu_avg = summary.process_cpu_avg if summary.process_cpu_avg > 0 else derived_process_cpu
    if summary.cpu_avg == 0.0 and process_cpu_avg > 0:
        warnings.append(
            "The measured loop was too short for a stable system-wide CPU utilization "
            "sample; process CPU time is reported instead."
        )

    if plan.device == "cuda" and summary.vram_peak_mb is None:
        warnings.append(
            "GPU telemetry (NVML) was unavailable during the benchmark; VRAM utilization "
            "could not be sampled."
        )
    if stats.samples_seen < plan.samples:
        warnings.append(
            f"Only {stats.samples_seen}/{plan.samples} planned samples were processed."
        )
    warnings.extend(stats.warnings)
    warnings = list(dict.fromkeys(warnings))

    samples = [
        BenchmarkSample(
            elapsed_seconds=record.elapsed_seconds,
            cpu_pct=record.cpu_pct,
            process_cpu_pct=record.process_cpu_pct,
            ram_mb=record.ram_mb,
            gpu_util_pct=record.gpu_util_pct,
            gpu_memory_used_mb=record.gpu_memory_used_mb,
        )
        for record in summary.samples
    ]

    result = BenchmarkResult(
        run_id=run_id,
        started_at=started_at,
        model_type=effective_config.model_type,
        device_requested=request.benchmark.device,
        device_used=plan.device,
        precision_requested=request.precision,
        precision_used=plan.precision,
        samples_used=stats.samples_seen,
        batch_size=plan.batch_size,
        steps_completed=stats.steps_completed,
        warmup_steps=plan.warmup_steps,
        duration_seconds=round(duration, 6),
        samples_per_second=round(samples_per_second, 3),
        steps_per_second=round(stats.steps_completed / duration, 3),
        avg_step_ms=round(avg_step_ms, 3),
        cpu_utilization_avg_pct=summary.cpu_avg,
        cpu_utilization_peak_pct=summary.cpu_peak,
        process_cpu_avg_pct=round(process_cpu_avg, 2),
        ram_baseline_mb=round(baseline_ram, 1),
        ram_peak_mb=round(ram_peak, 1),
        ram_delta_mb=round(ram_delta, 1),
        gpu_utilization_avg_pct=summary.gpu_avg,
        gpu_utilization_peak_pct=summary.gpu_peak,
        vram_baseline_mb=summary.vram_baseline_mb,
        vram_peak_mb=summary.vram_peak_mb,
        vram_delta_mb=(
            round(summary.vram_peak_mb - summary.vram_baseline_mb, 1)
            if summary.vram_peak_mb is not None and summary.vram_baseline_mb is not None
            else None
        ),
        training_flops_performed=int(compute.total_training_flops),
        measured_tflops=round(measured_tflops, 5),
        samples=samples,
        warnings=warnings,
    )
    comparison = _comparison(
        effective_config,
        hardware,
        plan.device,
        plan.precision,
        duration,
        samples_per_second,
        measured_tflops,
        compute.total_training_flops,
    )
    safety_notes.extend(
        [
            f"Measured loop duration: {duration:.3f}s; total call time including setup: "
            f"{total_wall:.3f}s.",
            "Short runs have higher variance. Repeat the benchmark before making large decisions.",
        ]
    )
    logger.info(
        "Benchmark %s completed in %.3fs (%.1f samples/s, prediction ratio %.2f)",
        run_id,
        duration,
        samples_per_second,
        comparison.time_ratio,
    )
    return BenchmarkResponse(
        result=result,
        comparison=comparison,
        effective_config=effective_config,
        safety_notes=safety_notes,
        disclaimer=DISCLAIMER,
    )
