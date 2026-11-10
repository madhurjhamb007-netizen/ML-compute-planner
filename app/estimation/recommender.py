"""Suitability scoring, bottleneck analysis, and resource recommendations."""

from __future__ import annotations

import math

from app.core.config import (
    ACTIVATION_BUFFER_FACTOR,
    BYTES_PER_MIB,
    CPU_RUNTIME_OVERHEAD_MIB,
    DATASET_COPY_OVERHEAD,
    GPU_CONTEXT_OVERHEAD_MIB,
    MEMORY_HEADROOM_FRACTION,
)
from app.estimation.memory import _dataset_elements_per_sample, _precision_bytes
from app.estimation.runtime import resolve_precision
from app.hardware.detector import primary_gpu
from app.schemas import (
    Bottleneck,
    ComputeEstimate,
    HardwareInfo,
    MemoryEstimate,
    ModelConfig,
    RecommendationSet,
    Suitability,
    TimeEstimate,
)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _capability_tuple(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def mixed_precision_support(hardware: HardwareInfo) -> tuple[bool, bool]:
    """Return (fp16_supported, bf16_supported) for the primary usable GPU."""
    gpu = primary_gpu(hardware)
    if gpu is None or not hardware.cuda_available:
        return False, False
    capability = _capability_tuple(gpu.compute_capability)
    if capability is None:
        return True, True  # unknown capability; modern CUDA build is assumed
    return capability >= (5, 3), capability >= (8, 0)


def build_suitability(
    config: ModelConfig,
    compute: ComputeEstimate,
    memory: MemoryEstimate,
    time: TimeEstimate,
    hardware: HardwareInfo,
) -> Suitability:
    reasons: list[str] = []
    cpu_score = 100
    if not memory.fits_cpu:
        cpu_score = 15
        reasons.append(
            f"Estimated CPU RAM ({memory.cpu_training_ram_mb:,.0f} MiB) exceeds usable "
            "available memory."
        )
    else:
        if compute.total_training_flops > 1e15:
            cpu_score -= 30
            reasons.append("Workload exceeds 1 PFLOP; CPU training would be very slow.")
        elif compute.total_training_flops > 1e14:
            cpu_score -= 18
            reasons.append("Workload is large for CPU-only training.")
        if config.model_type in ("cnn", "lstm", "transformer") and hardware.cuda_available:
            cpu_score -= 10
            reasons.append("Deep model would benefit from an available CUDA GPU.")
        if hardware.cpu.physical_cores < 4:
            cpu_score -= 10
            reasons.append("Few physical CPU cores limit parallel throughput.")
    if time.target_device == "cpu":
        if time.predicted_seconds > 7200:
            cpu_score -= 20
            reasons.append("Predicted CPU training time exceeds two hours.")
        elif time.predicted_seconds > 1800:
            cpu_score -= 10
            reasons.append("Predicted CPU training time exceeds 30 minutes.")
    cpu_score = int(max(0, min(100, cpu_score)))

    gpu = primary_gpu(hardware)
    if gpu is None:
        gpu_score = 0
        reasons.append("No NVIDIA GPU detected; GPU score is zero.")
    elif not hardware.cuda_available:
        gpu_score = 0
        reasons.append("GPU is visible but unusable by this PyTorch build.")
    elif config.model_type == "random_forest":
        gpu_score = 20
        reasons.append("Random Forest is implemented as a CPU workload; GPU adds no benefit.")
    elif not memory.fits_gpu:
        gpu_score = 15
        reasons.append("Estimated VRAM or host RAM does not fit on the detected GPU.")
    else:
        gpu_score = 92
        fp16_ok, bf16_ok = mixed_precision_support(hardware)
        if fp16_ok or bf16_ok:
            gpu_score += 5
            reasons.append("Mixed-precision support is available on the detected GPU.")
        if time.target_device == "cuda" and time.predicted_seconds > 3600:
            gpu_score -= 8
            reasons.append("Predicted GPU training time exceeds one hour.")
    gpu_score = int(max(0, min(100, gpu_score)))

    if memory.target_device == "cuda":
        summary = (
            "CUDA is the better fit: the estimated workload fits in VRAM and the GPU "
            "has substantially higher parallel throughput than the CPU."
        )
    elif config.model_type == "random_forest":
        summary = "CPU is the appropriate device: the tree ensemble is a CPU workload."
    elif gpu is not None and hardware.cuda_available:
        summary = (
            "CPU is recommended because the estimated GPU memory requirement does not "
            "fit or the workload is small enough that GPU overhead dominates."
        )
    else:
        summary = (
            "CPU-only execution is the available path. The estimate is feasible only "
            "if the memory and time predictions are acceptable for your deadline."
        )

    return Suitability(
        cpu_score=cpu_score,
        gpu_score=gpu_score,
        recommended_device=memory.target_device,
        cpu_fit=memory.fits_cpu,
        gpu_fit=memory.fits_gpu,
        summary=summary,
        reasons=reasons,
    )


def build_bottlenecks(
    config: ModelConfig,
    compute: ComputeEstimate,
    memory: MemoryEstimate,
    time: TimeEstimate,
    hardware: HardwareInfo,
) -> list[Bottleneck]:
    items: list[Bottleneck] = []
    if not memory.fits_target:
        items.append(
            Bottleneck(
                category="memory",
                severity="high",
                message=(
                    f"Estimated target memory ({memory.target_required_mb:,.0f} MiB) "
                    f"exceeds the usable budget of {memory.available_target_mb:,.0f} MiB. "
                    "Reduce batch size or model dimensions before running."
                ),
            )
        )
    elif memory.target_utilization_pct >= 85:
        items.append(
            Bottleneck(
                category="memory",
                severity="medium",
                message=(
                    f"Memory utilization is estimated at {memory.target_utilization_pct:.0f}% "
                    "of available capacity, leaving little room for fragmentation."
                ),
            )
        )

    if config.model_type == "random_forest":
        items.append(
            Bottleneck(
                category="compute",
                severity="medium",
                message=(
                    "Tree construction is CPU-bound; prediction quality and runtime depend "
                    "on split strategy, max_depth, and feature sampling."
                ),
            )
        )
    elif config.model_type in ("lstm", "transformer") and time.target_device == "cpu":
        items.append(
            Bottleneck(
                category="compute",
                severity="medium",
                message=(
                    "Sequential/recurrent or attention-heavy training on CPU is usually "
                    "the dominant bottleneck; a GPU is strongly preferred for long runs."
                ),
            )
        )

    if config.model_type == "transformer" and config.sequence_length >= 256:
        items.append(
            Bottleneck(
                category="memory",
                severity="high" if config.sequence_length >= 1024 else "medium",
                message=(
                    "Transformer attention activations scale with sequence_length². "
                    "Consider shorter sequences, gradient checkpointing, or a larger GPU."
                ),
            )
        )

    if config.batch_size < 16:
        items.append(
            Bottleneck(
                category="batch_size",
                severity="medium",
                message="Batch size below 16 usually underutilizes CPU/GPU kernels.",
            )
        )
    if time.target_device == "cuda" and config.batch_size > 512:
        items.append(
            Bottleneck(
                category="batch_size",
                severity="low",
                message=(
                    "Very large batches can reduce generalization and increase memory "
                    "pressure; consider gradient accumulation instead."
                ),
            )
        )

    if not hardware.cuda_available and config.model_type in ("cnn", "lstm", "transformer"):
        items.append(
            Bottleneck(
                category="device",
                severity="high",
                message=(
                    "No usable CUDA GPU is available for a deep model. Expect long "
                    "training times; use a smaller configuration for experimentation."
                ),
            )
        )

    if config.dataset_size > 100_000 and config.batch_size < 64:
        items.append(
            Bottleneck(
                category="data_loading",
                severity="low",
                message=(
                    "Large dataset with small batches may become input-pipeline bound. "
                    "Use more DataLoader workers and precompute features where possible."
                ),
            )
        )
    if hardware.cpu.physical_cores <= 4 and config.model_type != "random_forest":
        items.append(
            Bottleneck(
                category="cpu_cores",
                severity="medium",
                message=(
                    "Only a few physical cores are available; CPU training throughput "
                    "will be limited and thread oversubscription should be avoided."
                ),
            )
        )
    if config.precision != "fp32":
        resolution = resolve_precision(config, hardware, time.target_device)
        if resolution.effective != config.precision:
            items.append(
                Bottleneck(
                    category="precision",
                    severity="low",
                    message=(
                        f"Requested {config.precision} is not used; effective precision is "
                        f"{resolution.effective}. See the estimation warnings."
                    ),
                )
            )
    if not items:
        items.append(
            Bottleneck(
                category="compute",
                severity="low",
                message="No dominant bottleneck detected from the current configuration.",
            )
        )
    items.sort(key=lambda item: _SEVERITY_ORDER[item.severity])
    return items


def _max_safe_batch(
    config: ModelConfig,
    compute: ComputeEstimate,
    memory: MemoryEstimate,
    hardware: HardwareInfo,
) -> int:
    if config.model_type == "random_forest":
        return min(512, config.dataset_size)
    precision = resolve_precision(config, hardware, memory.target_device).effective
    act_bytes = _precision_bytes(precision)
    activation_per_sample = (
        compute.activation_elements * act_bytes * ACTIVATION_BUFFER_FACTOR / BYTES_PER_MIB
    )
    input_per_sample = (
        _dataset_elements_per_sample(config) * 4.0 * 1.1 / BYTES_PER_MIB
    )
    variable_per_sample = max(1e-6, activation_per_sample + input_per_sample)
    params = compute.parameter_count
    optimizer_bytes = 8.0 if config.optimizer in ("adam", "adamw") else 4.0
    model_fixed = params * (4.0 + 4.0 + optimizer_bytes) / BYTES_PER_MIB
    budget = memory.available_target_mb * MEMORY_HEADROOM_FRACTION
    if memory.target_device == "cuda":
        fixed = model_fixed + GPU_CONTEXT_OVERHEAD_MIB
    else:
        dataset_mb = (
            config.dataset_size
            * _dataset_elements_per_sample(config)
            * 4.0
            * DATASET_COPY_OVERHEAD
            / BYTES_PER_MIB
        )
        fixed = model_fixed + dataset_mb + CPU_RUNTIME_OVERHEAD_MIB
    available_for_batch = max(0.0, budget - fixed)
    max_batch = math.floor(available_for_batch / variable_per_sample)
    return int(max(1, min(4096, max_batch)))


def build_recommendations(
    config: ModelConfig,
    compute: ComputeEstimate,
    memory: MemoryEstimate,
    time: TimeEstimate,
    hardware: HardwareInfo,
) -> RecommendationSet:
    cpu = hardware.cpu
    cpu_threads = max(1, min(cpu.recommended_threads, cpu.logical_threads))
    workers = max(1, min(4, cpu.physical_cores // 2)) if cpu.physical_cores > 2 else 1
    target = memory.target_device
    max_safe_batch = _max_safe_batch(config, compute, memory, hardware)
    preferred_batch = 256 if target == "cuda" else 64
    recommended_batch = max(1, min(max_safe_batch, config.dataset_size, preferred_batch))
    if config.model_type == "random_forest":
        recommended_batch = min(512, config.dataset_size)
        workers = max(1, min(cpu.physical_cores, 8))
        max_safe_batch = recommended_batch

    fp16_ok, bf16_ok = mixed_precision_support(hardware)
    mixed_available = fp16_ok or bf16_ok
    if target == "cuda" and mixed_available:
        recommended_precision = "bf16" if bf16_ok else "fp16"
    elif target == "cuda":
        recommended_precision = "fp32"
    else:
        recommended_precision = "fp32"

    memory_optimizations: list[str] = []
    general_advice: list[str] = []
    notes: list[str] = []

    if not memory.fits_target:
        memory_optimizations.extend(
            [
                "Enable gradient checkpointing to trade extra compute for lower activation memory.",
                "Use gradient accumulation with a smaller micro-batch to reach the desired effective batch.",
                "Reduce hidden size, layers, sequence length, or image resolution before starting.",
                "Free the dataset cache or stream batches from disk if host RAM is the constraint.",
            ]
        )
    elif memory.target_utilization_pct >= 85:
        memory_optimizations.append(
            "Memory headroom is tight; keep batch size fixed and monitor peak memory during early steps."
        )
    activation_share = 0.0
    breakdown = memory.gpu_breakdown if target == "cuda" else memory.cpu_breakdown
    if breakdown:
        total = sum(item.mb for item in breakdown) or 1.0
        activation_item = next((item for item in breakdown if item.name.startswith("Activations")), None)
        if activation_item:
            activation_share = activation_item.mb / total
            if activation_share > 0.30:
                memory_optimizations.append(
                    "Activation memory is a large share of the footprint; gradient checkpointing may help."
                )
    optimizer_item = next((item for item in breakdown if item.name.startswith("Optimizer")), None)
    if optimizer_item and target == "cuda" and optimizer_item.mb > 0.20 * max(1.0, sum(i.mb for i in breakdown)):
        memory_optimizations.append(
            "Optimizer state is significant; SGD with momentum or an 8-bit optimizer can reduce VRAM."
        )
    if config.batch_size > max_safe_batch:
        memory_optimizations.append(
            f"Configured batch size {config.batch_size} exceeds the estimated safe batch "
            f"of {max_safe_batch}; reduce it or use gradient accumulation."
        )
    if config.model_type == "cnn":
        general_advice.append("For CNN training, channels_last plus AMP often improves GPU throughput.")
    if config.model_type == "transformer":
        general_advice.append(
            "Use scaled_dot_product_attention / fused attention kernels when available and "
            "activate gradient checkpointing for long sequences."
        )
    if config.model_type == "random_forest":
        general_advice.extend(
            [
                "Limit max_depth and use sqrt(features) for split candidates to control runtime.",
                "Parallelize independent trees across CPU cores; avoid nested parallelism.",
                "Consider histogram-based gradient boosting for large tabular datasets.",
            ]
        )
    if target == "cuda":
        general_advice.extend(
            [
                "Use pin_memory=True and non_blocking transfers in the DataLoader.",
                "Warm up for a few steps, then measure with CUDA events or synchronized timers.",
                "Monitor GPU temperature and power limits; sustained clocks may be lower than peak.",
            ]
        )
    else:
        general_advice.extend(
            [
                f"Set torch.set_num_threads({cpu_threads}) to avoid CPU oversubscription.",
                "Measure a small run before launching a long CPU training job.",
                "Use a GPU when available for CNNs, LSTMs, and Transformers.",
            ]
        )
    general_advice.append(
        "Use the benchmark section to calibrate predictions on your own hardware and workload."
    )
    if time.confidence == "low":
        notes.append("Prediction confidence is low; calibrate with the benchmark before committing resources.")
    if workers > 0 and hardware.os.lower().startswith("windows"):
        notes.append(
            "On Windows, DataLoader worker processes require a __main__ guard in standalone "
            "scripts. The built-in benchmark does not use DataLoader workers."
        )
    if config.precision != recommended_precision:
        notes.append(
            f"Requested precision {config.precision}; recommended precision for this path is "
            f"{recommended_precision}."
        )

    return RecommendationSet(
        recommended_device=target,
        cpu_threads=cpu_threads,
        dataloader_workers=workers,
        batch_size=recommended_batch,
        max_safe_batch_size=max_safe_batch,
        precision=recommended_precision,
        mixed_precision_available=mixed_available,
        memory_optimizations=list(dict.fromkeys(memory_optimizations)),
        general_advice=list(dict.fromkeys(general_advice)),
        notes=notes,
    )
