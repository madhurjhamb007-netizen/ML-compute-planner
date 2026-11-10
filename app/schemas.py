"""Pydantic schemas shared by the API, estimators, benchmark, and dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelType = Literal[
    "linear_regression",
    "random_forest",
    "mlp",
    "cnn",
    "lstm",
    "transformer",
]
Precision = Literal["fp32", "fp16", "bf16"]
DevicePreference = Literal["auto", "cpu", "cuda"]
ResolvedDevice = Literal["cpu", "cuda"]
OptimizerName = Literal["adam", "adamw", "sgd"]
Severity = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium"]


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


class CPUInfo(APIModel):
    name: str
    architecture: str
    physical_cores: int
    logical_threads: int
    current_frequency_mhz: float | None = None
    max_frequency_mhz: float | None = None
    utilization_pct: float
    per_core_utilization_pct: list[float] = Field(default_factory=list)
    recommended_threads: int
    estimated_fp32_gflops: float
    notes: list[str] = Field(default_factory=list)


class RAMInfo(APIModel):
    total_mb: float
    available_mb: float
    used_mb: float
    utilization_pct: float


class GPUInfo(APIModel):
    index: int
    name: str
    total_memory_mb: float
    used_memory_mb: float | None = None
    free_memory_mb: float | None = None
    utilization_pct: float | None = None
    memory_utilization_pct: float | None = None
    compute_capability: str | None = None
    multiprocessors: int | None = None
    clock_mhz: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    driver_version: str | None = None
    cuda_available: bool = False
    source: str
    notes: list[str] = Field(default_factory=list)


class HardwareInfo(APIModel):
    hostname: str
    os: str
    python_version: str
    detected_at: datetime
    detection_seconds: float
    cpu: CPUInfo
    ram: RAMInfo
    gpus: list[GPUInfo] = Field(default_factory=list)
    cuda_available: bool = False
    torch_available: bool = False
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    nvidia_driver_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model configuration and registry
# ---------------------------------------------------------------------------


class ModelConfig(APIModel):
    model_type: ModelType = "mlp"
    dataset_size: int = Field(default=5000, ge=1, le=10_000_000)
    features: int = Field(default=32, ge=1, le=8192)
    batch_size: int = Field(default=64, ge=1, le=4096)
    epochs: int = Field(default=10, ge=1, le=10_000)
    precision: Precision = "fp32"
    device_preference: DevicePreference = "auto"
    optimizer: OptimizerName = "adam"

    hidden_size: int = Field(default=128, ge=1, le=8192)
    num_layers: int = Field(default=2, ge=1, le=64)
    num_heads: int = Field(default=4, ge=1, le=64)
    sequence_length: int = Field(default=64, ge=1, le=8192)
    image_size: int = Field(default=32, ge=16, le=1024)
    in_channels: int = Field(default=3, ge=1, le=16)
    base_channels: int = Field(default=32, ge=1, le=1024)
    num_classes: int = Field(default=10, ge=2, le=10000)
    n_estimators: int = Field(default=100, ge=1, le=2000)
    max_depth: int = Field(default=10, ge=1, le=64)

    @model_validator(mode="after")
    def validate_model_specific(self) -> "ModelConfig":
        if self.model_type == "transformer":
            if self.num_heads > self.hidden_size:
                raise ValueError("num_heads cannot exceed hidden_size")
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_heads for a Transformer"
                )
        if self.model_type == "cnn" and self.image_size < 16:
            raise ValueError("image_size must be at least 16 for the CNN benchmark")
        return self


class ModelInfo(APIModel):
    key: ModelType
    display_name: str
    family: str
    description: str
    supports_gpu: bool
    gpu_friendly: bool
    default_config: ModelConfig
    notes: str


# ---------------------------------------------------------------------------
# Compute estimation
# ---------------------------------------------------------------------------


class LayerBreakdown(APIModel):
    name: str
    parameters: int
    forward_flops: int
    activation_elements: int


class ComputeEstimate(APIModel):
    parameter_count: int
    trainable_parameters: int
    forward_flops_per_sample: int
    backward_flops_per_sample: int
    training_flops_per_sample: int
    total_forward_flops: int
    total_training_flops: int
    optimizer_steps: int
    activation_elements: int
    breakdown: list[LayerBreakdown]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


class MemoryItem(APIModel):
    name: str
    mb: float


class MemoryEstimate(APIModel):
    cpu_training_ram_mb: float
    gpu_training_vram_mb: float
    gpu_host_ram_mb: float
    gpu_inference_vram_mb: float
    cpu_breakdown: list[MemoryItem]
    gpu_breakdown: list[MemoryItem]
    host_breakdown: list[MemoryItem]
    target_device: ResolvedDevice
    target_required_mb: float
    available_target_mb: float
    target_utilization_pct: float
    fits_cpu: bool
    fits_gpu: bool | None
    fits_target: bool
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Time and suitability
# ---------------------------------------------------------------------------


class TimeEstimate(APIModel):
    target_device: ResolvedDevice
    precision_used: Precision
    predicted_seconds: float
    predicted_low_seconds: float
    predicted_high_seconds: float
    predicted_hours: float
    throughput_samples_per_second: float
    effective_tflops: float
    compute_seconds: float
    overhead_seconds: float
    confidence: Confidence
    bottleneck_primary: str
    notes: list[str] = Field(default_factory=list)


class Bottleneck(APIModel):
    category: Literal[
        "memory", "compute", "data_loading", "batch_size", "device", "precision", "cpu_cores"
    ]
    severity: Severity
    message: str


class Suitability(APIModel):
    cpu_score: int
    gpu_score: int
    recommended_device: ResolvedDevice
    cpu_fit: bool
    gpu_fit: bool | None
    summary: str
    reasons: list[str] = Field(default_factory=list)


class RecommendationSet(APIModel):
    recommended_device: ResolvedDevice
    cpu_threads: int
    dataloader_workers: int
    batch_size: int
    max_safe_batch_size: int
    precision: Precision
    mixed_precision_available: bool
    memory_optimizations: list[str]
    general_advice: list[str]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full estimate response
# ---------------------------------------------------------------------------


class EstimateResponse(APIModel):
    config: ModelConfig
    hardware: HardwareInfo
    compute: ComputeEstimate
    memory: MemoryEstimate
    time: TimeEstimate
    recommendations: RecommendationSet
    suitability: Suitability
    bottlenecks: list[Bottleneck]
    assumptions: list[str]
    warnings: list[str]
    disclaimer: str
    estimation_seconds: float


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkOptions(APIModel):
    device: DevicePreference = "auto"
    steps: int = Field(default=20, ge=1, le=100)
    warmup_steps: int = Field(default=2, ge=0, le=5)
    samples: int | None = Field(default=None, ge=1, le=2048)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    max_seconds: float = Field(default=15.0, ge=1.0, le=15.0)


class BenchmarkRequest(ModelConfig):
    benchmark: BenchmarkOptions = Field(default_factory=BenchmarkOptions)


class BenchmarkSample(APIModel):
    elapsed_seconds: float
    cpu_pct: float
    process_cpu_pct: float
    ram_mb: float
    gpu_util_pct: float | None = None
    gpu_memory_used_mb: float | None = None


class BenchmarkResult(APIModel):
    run_id: str
    started_at: datetime
    model_type: ModelType
    device_requested: DevicePreference
    device_used: ResolvedDevice
    precision_requested: Precision
    precision_used: Precision
    samples_used: int
    batch_size: int
    steps_completed: int
    warmup_steps: int
    duration_seconds: float
    samples_per_second: float
    steps_per_second: float
    avg_step_ms: float
    cpu_utilization_avg_pct: float
    cpu_utilization_peak_pct: float
    process_cpu_avg_pct: float
    ram_baseline_mb: float
    ram_peak_mb: float
    ram_delta_mb: float
    gpu_utilization_avg_pct: float | None = None
    gpu_utilization_peak_pct: float | None = None
    vram_baseline_mb: float | None = None
    vram_peak_mb: float | None = None
    vram_delta_mb: float | None = None
    training_flops_performed: int
    measured_tflops: float
    samples: list[BenchmarkSample]
    warnings: list[str] = Field(default_factory=list)


class BenchmarkComparison(APIModel):
    predicted_seconds: float
    predicted_low_seconds: float
    predicted_high_seconds: float
    measured_seconds: float
    time_ratio: float
    time_error_pct: float
    predicted_samples_per_second: float
    measured_samples_per_second: float
    throughput_ratio: float
    predicted_tflops: float
    measured_tflops: float
    within_expected_range: bool
    interpretation: str


class BenchmarkResponse(APIModel):
    result: BenchmarkResult
    comparison: BenchmarkComparison
    effective_config: ModelConfig
    safety_notes: list[str] = Field(default_factory=list)
    disclaimer: str


# ---------------------------------------------------------------------------
# Small utility responses
# ---------------------------------------------------------------------------


class HealthResponse(APIModel):
    status: str
    app: str
    version: str
    timestamp: datetime


class RootResponse(APIModel):
    app: str
    version: str
    docs_url: str
    dashboard_hint: str
