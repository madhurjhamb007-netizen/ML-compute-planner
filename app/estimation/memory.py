"""Training and inference memory estimation.

All values are MiB. The estimator deliberately adds explicit runtime overheads
(CUDA context, Python/Torch runtime, activation buffering) and labels the result
as a planning estimate rather than a measured allocation.
"""

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
from app.hardware.detector import primary_gpu
from app.schemas import (
    ComputeEstimate,
    HardwareInfo,
    MemoryEstimate,
    MemoryItem,
    ModelConfig,
    Precision,
)
from app.estimation.runtime import resolve_precision


def _mib(value_bytes: float) -> float:
    return round(max(0.0, value_bytes) / BYTES_PER_MIB, 1)


def _precision_bytes(precision: Precision) -> float:
    return 2.0 if precision in ("fp16", "bf16") else 4.0


def _dataset_elements_per_sample(config: ModelConfig) -> int:
    if config.model_type == "cnn":
        return config.in_channels * config.image_size * config.image_size
    if config.model_type in ("lstm", "transformer"):
        return config.sequence_length * config.features
    return config.features


def _optimizer_bytes(model_type: str, optimizer: str, params: int) -> float:
    if model_type == "random_forest":
        return 0.0
    if optimizer in ("adam", "adamw"):
        return params * 8.0  # first and second moments (fp32 each)
    return params * 4.0  # SGD momentum buffer


def estimate_memory(
    config: ModelConfig, compute: ComputeEstimate, hardware: HardwareInfo
) -> MemoryEstimate:
    notes: list[str] = []
    cpu_precision = resolve_precision(config, hardware, "cpu").effective
    gpu_precision = resolve_precision(config, hardware, "cuda").effective
    gpu = primary_gpu(hardware)

    dataset_mb = _mib(
        config.dataset_size
        * _dataset_elements_per_sample(config)
        * 4.0
        * DATASET_COPY_OVERHEAD
    )
    input_elements = _dataset_elements_per_sample(config)
    input_batch_mb = _mib(input_elements * config.batch_size * 4.0 * 1.1)

    if config.model_type == "random_forest":
        forest_mb = _mib(compute.parameter_count * 8.0)
        working_mb = round(dataset_mb * 0.25 + 16.0, 1)
        cpu_items = [
            MemoryItem(name="Dataset (host)", mb=dataset_mb),
            MemoryItem(name="Forest nodes", mb=forest_mb),
            MemoryItem(name="Tree-building workspace", mb=working_mb),
            MemoryItem(name="Python/runtime overhead", mb=CPU_RUNTIME_OVERHEAD_MIB),
        ]
        cpu_total = round(sum(item.mb for item in cpu_items), 1)
        available_ram = hardware.ram.available_mb
        fits_cpu = available_ram > 0 and cpu_total <= available_ram * MEMORY_HEADROOM_FRACTION
        notes.extend(
            [
                "Random Forest uses CPU memory only; GPU VRAM is not applicable.",
                "Dataset memory assumes the feature matrix is materialized in RAM.",
            ]
        )
        target: str = "cpu"
        return MemoryEstimate(
            cpu_training_ram_mb=cpu_total,
            gpu_training_vram_mb=0.0,
            gpu_host_ram_mb=0.0,
            gpu_inference_vram_mb=0.0,
            cpu_breakdown=cpu_items,
            gpu_breakdown=[],
            host_breakdown=[],
            target_device=target,  # type: ignore[arg-type]
            target_required_mb=cpu_total,
            available_target_mb=available_ram,
            target_utilization_pct=(
                round(cpu_total / available_ram * 100.0, 1) if available_ram else 100.0
            ),
            fits_cpu=fits_cpu,
            fits_gpu=None,
            fits_target=fits_cpu,
            notes=notes,
        )

    params = compute.parameter_count
    weights_mb = _mib(params * 4.0)
    gradients_mb = _mib(params * 4.0)
    optimizer_mb = _mib(_optimizer_bytes(config.model_type, config.optimizer, params))
    activation_mb_cpu = _mib(
        compute.activation_elements
        * config.batch_size
        * _precision_bytes(cpu_precision)
        * ACTIVATION_BUFFER_FACTOR
    )
    activation_mb_gpu = _mib(
        compute.activation_elements
        * config.batch_size
        * _precision_bytes(gpu_precision)
        * ACTIVATION_BUFFER_FACTOR
    )

    cpu_payload = weights_mb + gradients_mb + optimizer_mb + activation_mb_cpu + input_batch_mb
    cpu_overhead = round(CPU_RUNTIME_OVERHEAD_MIB + cpu_payload * 0.15, 1)
    cpu_items = [
        MemoryItem(name="Model weights", mb=weights_mb),
        MemoryItem(name="Gradients", mb=gradients_mb),
        MemoryItem(name="Optimizer state", mb=optimizer_mb),
        MemoryItem(name="Activations (batch)", mb=activation_mb_cpu),
        MemoryItem(name="Input batch", mb=input_batch_mb),
        MemoryItem(name="Dataset (host)", mb=dataset_mb),
        MemoryItem(name="Runtime + buffering", mb=cpu_overhead),
    ]
    cpu_total = round(sum(item.mb for item in cpu_items), 1)

    gpu_payload = weights_mb + gradients_mb + optimizer_mb + activation_mb_gpu + input_batch_mb
    gpu_workspace = round(GPU_CONTEXT_OVERHEAD_MIB + gpu_payload * 0.10, 1)
    gpu_items = [
        MemoryItem(name="Model weights (fp32 master)", mb=weights_mb),
        MemoryItem(name="Gradients", mb=gradients_mb),
        MemoryItem(name="Optimizer state", mb=optimizer_mb),
        MemoryItem(name="Activations (batch)", mb=activation_mb_gpu),
        MemoryItem(name="Input batch", mb=input_batch_mb),
        MemoryItem(name="CUDA context + workspace", mb=gpu_workspace),
    ]
    gpu_total = round(sum(item.mb for item in gpu_items), 1)

    host_items = [
        MemoryItem(name="Dataset (host)", mb=dataset_mb),
        MemoryItem(name="Loader/input buffers", mb=round(input_batch_mb * 2.0, 1)),
        MemoryItem(name="Python/runtime overhead", mb=CPU_RUNTIME_OVERHEAD_MIB),
    ]
    host_total = round(sum(item.mb for item in host_items), 1)

    inference_weight_bytes = 2.0 if gpu_precision in ("fp16", "bf16") else 4.0
    inference_vram = _mib(
        params * inference_weight_bytes
        + compute.activation_elements
        * config.batch_size
        * _precision_bytes(gpu_precision)
        * 0.5
        + input_elements * config.batch_size * 4.0
    ) + GPU_CONTEXT_OVERHEAD_MIB
    inference_vram = round(inference_vram, 1)

    available_ram = hardware.ram.available_mb
    fits_cpu = available_ram > 0 and cpu_total <= available_ram * MEMORY_HEADROOM_FRACTION

    gpu_fit: bool | None
    free_vram = 0.0
    if gpu is None or not hardware.cuda_available:
        gpu_fit = None
        if gpu is not None and not hardware.cuda_available:
            notes.append(
                "A GPU is physically present, but CUDA is unavailable to PyTorch; "
                "GPU fit cannot be verified and GPU execution is disabled."
            )
        else:
            notes.append(
                "No usable CUDA GPU: VRAM numbers are hypothetical requirements that "
                "can help choose a future GPU. CPU execution is used."
            )
    else:
        free_vram = float(gpu.free_memory_mb or gpu.total_memory_mb)
        gpu_fit = (
            gpu_total <= free_vram * MEMORY_HEADROOM_FRACTION
            and host_total <= available_ram * MEMORY_HEADROOM_FRACTION
        )
        if gpu_total > free_vram * MEMORY_HEADROOM_FRACTION:
            notes.append(
                f"Estimated VRAM ({gpu_total:,.0f} MiB) exceeds the usable free VRAM "
                f"({free_vram * MEMORY_HEADROOM_FRACTION:,.0f} MiB)."
            )

    if config.device_preference == "cpu":
        target = "cpu"
        if hardware.cuda_available:
            notes.append("CPU was explicitly requested; GPU capacity is reported separately.")
    elif config.device_preference == "cuda":
        if gpu_fit:
            target = "cuda"
        else:
            target = "cpu"
            notes.append("CUDA was requested but is unavailable or the workload does not fit; using CPU.")
    else:
        target = "cuda" if gpu_fit else "cpu"
        if gpu_fit:
            notes.append("Auto device selection chose CUDA because the estimated workload fits.")
        elif hardware.cuda_available:
            notes.append("Auto device selection chose CPU because the GPU memory fit failed.")

    if target == "cuda" and gpu is not None:
        available_target = float(gpu.free_memory_mb or gpu.total_memory_mb)
        required = gpu_total
        fits_target = bool(gpu_fit) and host_total <= available_ram * MEMORY_HEADROOM_FRACTION
        if host_total > available_ram * MEMORY_HEADROOM_FRACTION:
            notes.append("Host RAM is insufficient for the dataset/loader buffers at this batch size.")
    else:
        available_target = available_ram
        required = cpu_total
        fits_target = fits_cpu

    utilization = round(required / available_target * 100.0, 1) if available_target else 100.0
    notes.extend(
        [
            "Activation memory is an analytical approximation; exact autograd storage "
            "depends on PyTorch kernels, fusion, and checkpointing.",
            "Dataset memory assumes the full dataset is materialized in host RAM; "
            "streaming or on-disk pipelines reduce this.",
            "CUDA context overhead is a generic 512 MiB allowance, not a device lookup.",
        ]
    )
    return MemoryEstimate(
        cpu_training_ram_mb=cpu_total,
        gpu_training_vram_mb=gpu_total,
        gpu_host_ram_mb=host_total,
        gpu_inference_vram_mb=inference_vram,
        cpu_breakdown=cpu_items,
        gpu_breakdown=gpu_items,
        host_breakdown=host_items,
        target_device=target,  # type: ignore[arg-type]
        target_required_mb=required,
        available_target_mb=available_target,
        target_utilization_pct=utilization,
        fits_cpu=fits_cpu,
        fits_gpu=gpu_fit,
        fits_target=fits_target,
        notes=notes,
    )


def estimated_activation_mb(config: ModelConfig, compute: ComputeEstimate) -> float:
    """Per-batch activation estimate used by benchmark safety checks."""
    return _mib(
        compute.activation_elements
        * config.batch_size
        * _precision_bytes(config.precision)
        * ACTIVATION_BUFFER_FACTOR
    )


def format_mib(value_mb: float) -> str:
    if value_mb >= 1024:
        return f"{value_mb / 1024:.2f} GiB"
    return f"{value_mb:,.0f} MiB"
