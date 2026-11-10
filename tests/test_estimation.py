"""Analytical estimation, memory, time, and recommendation tests."""

from __future__ import annotations

import math

import pytest

from app.estimation.flops import estimate_compute
from app.estimation.memory import estimate_memory
from app.estimation.recommender import (
    build_bottlenecks,
    build_recommendations,
    build_suitability,
    mixed_precision_support,
)
from app.estimation.runtime import resolve_precision
from app.estimation.time_predictor import (
    predict_benchmark_duration,
    predict_training_time,
)
from app.models.architectures import build_torch_model
from app.models.registry import default_config
from app.schemas import ModelConfig
from app.services.planner import estimate_plan


def _config(model_type: str, **updates: int | str) -> ModelConfig:
    return default_config(model_type).model_copy(update=updates)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parameters and FLOPs
# ---------------------------------------------------------------------------

NEURAL_CASES = [
    ("linear_regression", {"features": 17}),
    ("mlp", {"features": 17, "hidden_size": 33, "num_layers": 3, "num_classes": 5}),
    (
        "cnn",
        {
            "in_channels": 1,
            "image_size": 28,
            "base_channels": 8,
            "hidden_size": 48,
            "num_classes": 4,
        },
    ),
    ("lstm", {"features": 11, "sequence_length": 23, "hidden_size": 19, "num_layers": 3, "num_classes": 3}),
    (
        "transformer",
        {
            "features": 12,
            "sequence_length": 31,
            "hidden_size": 32,
            "num_heads": 4,
            "num_layers": 3,
            "num_classes": 6,
        },
    ),
]


@pytest.mark.parametrize(("model_type", "updates"), NEURAL_CASES)
def test_parameter_counts_match_real_pytorch_modules(model_type: str, updates: dict) -> None:
    pytest.importorskip("torch")
    config = _config(model_type, **updates)
    model = build_torch_model(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    predicted = estimate_compute(config).parameter_count
    assert predicted == actual


def test_mlp_forward_flops_are_exact_for_the_implemented_network() -> None:
    config = _config(
        "mlp", features=20, hidden_size=40, num_layers=3, num_classes=7, dataset_size=100, epochs=2
    )
    compute = estimate_compute(config)
    expected_forward = 2 * (
        config.features * config.hidden_size
        + (config.num_layers - 1) * config.hidden_size**2
        + config.hidden_size * config.num_classes
    )
    assert compute.forward_flops_per_sample == expected_forward
    assert compute.training_flops_per_sample == 3 * expected_forward
    assert compute.total_forward_flops == expected_forward * 100 * 2
    assert compute.total_training_flops == 3 * expected_forward * 100 * 2
    assert compute.optimizer_steps == math.ceil(100 / config.batch_size) * 2
    assert compute.activation_elements == sum(
        item.activation_elements for item in compute.breakdown
    )
    assert compute.parameter_count == sum(item.parameters for item in compute.breakdown)


def test_cnn_flops_scale_with_image_and_channel_width() -> None:
    small = estimate_compute(_config("cnn", image_size=32, base_channels=8))
    large_image = estimate_compute(_config("cnn", image_size=64, base_channels=8))
    wide = estimate_compute(_config("cnn", image_size=32, base_channels=16))
    assert large_image.forward_flops_per_sample > small.forward_flops_per_sample
    assert wide.forward_flops_per_sample > small.forward_flops_per_sample
    assert large_image.parameter_count == small.parameter_count  # parameter count uses pooled features


def test_transformer_attention_cost_scales_superlinearly_with_sequence_length() -> None:
    short = estimate_compute(_config("transformer", sequence_length=64))
    long = estimate_compute(_config("transformer", sequence_length=128))
    assert long.forward_flops_per_sample > 2 * short.forward_flops_per_sample
    assert long.parameter_count > short.parameter_count  # the learned positional table grows


def test_random_forest_is_a_tree_workload_not_a_gradient_model() -> None:
    config = _config("random_forest", dataset_size=500, n_estimators=20, max_depth=8)
    compute = estimate_compute(config)
    assert compute.parameter_count > 0
    assert compute.optimizer_steps == 0
    assert compute.backward_flops_per_sample == 0
    assert compute.training_flops_per_sample > 0
    assert compute.total_forward_flops == compute.forward_flops_per_sample * config.dataset_size
    assert compute.total_training_flops > 0
    assert any("epochs" in note.lower() for note in compute.notes)


def test_random_forest_ignores_epochs_and_caps_effective_depth() -> None:
    base = _config(
        "random_forest", dataset_size=16, n_estimators=5, max_depth=64, epochs=1
    )
    many_epochs = base.model_copy(update={"epochs": 50})
    assert estimate_compute(base).total_training_flops == estimate_compute(many_epochs).total_training_flops
    assert any("limited to 4" in note for note in estimate_compute(base).notes)


def test_doubling_dataset_doubles_training_work() -> None:
    base = _config("mlp", dataset_size=1000, epochs=3)
    doubled = base.model_copy(update={"dataset_size": 2000})
    assert estimate_compute(doubled).total_training_flops == 2 * estimate_compute(base).total_training_flops


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _item_mb(memory, name_prefix: str) -> float:
    item = next(item for item in memory.cpu_breakdown + memory.gpu_breakdown if item.name.startswith(name_prefix))
    return item.mb


def test_cpu_only_memory_keeps_gpu_requirements_hypothetical(cpu_only_hardware) -> None:
    config = _config("mlp")
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, cpu_only_hardware)

    assert memory.target_device == "cpu"
    assert memory.fits_gpu is None
    assert memory.gpu_training_vram_mb > 0
    assert memory.gpu_inference_vram_mb > 0
    assert memory.gpu_host_ram_mb > 0
    assert memory.fits_target is memory.fits_cpu
    assert all(item.mb >= 0 for item in memory.cpu_breakdown)
    assert memory.cpu_training_ram_mb == pytest.approx(
        sum(item.mb for item in memory.cpu_breakdown), abs=0.2
    )


def test_adam_uses_more_optimizer_memory_than_sgd(cpu_only_hardware) -> None:
    compute = estimate_compute(_config("mlp"))
    adam = estimate_memory(_config("mlp", optimizer="adam"), compute, cpu_only_hardware)
    sgd = estimate_memory(_config("mlp", optimizer="sgd"), compute, cpu_only_hardware)
    assert _item_mb(adam, "Optimizer") > _item_mb(sgd, "Optimizer")


def test_activation_memory_scales_with_batch_size(cpu_only_hardware) -> None:
    small_config = _config("transformer", batch_size=32)
    large_config = small_config.model_copy(update={"batch_size": 64})
    compute = estimate_compute(small_config)
    small = estimate_memory(small_config, compute, cpu_only_hardware)
    large = estimate_memory(large_config, compute, cpu_only_hardware)
    assert _item_mb(large, "Activations") == pytest.approx(2 * _item_mb(small, "Activations"), rel=0.02)


def test_gpu_fit_and_auto_target_on_realistic_gpu(gpu_hardware) -> None:
    config = _config("mlp", batch_size=64)
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, gpu_hardware)

    assert memory.fits_gpu is True
    assert memory.fits_cpu is True
    assert memory.target_device == "cuda"
    assert memory.available_target_mb == gpu_hardware.gpus[0].free_memory_mb
    assert memory.target_utilization_pct < 100
    assert memory.fits_target is True
    assert memory.gpu_training_vram_mb > 0


def test_explicit_cpu_preference_overrides_gpu(gpu_hardware) -> None:
    config = _config("mlp", device_preference="cpu")
    memory = estimate_memory(config, estimate_compute(config), gpu_hardware)
    assert memory.target_device == "cpu"
    assert memory.fits_gpu is True  # capacity is still reported
    assert any("explicitly requested" in note for note in memory.notes)


def test_cuda_request_falls_back_to_cpu_with_explanation(cpu_only_hardware) -> None:
    config = _config("mlp", device_preference="cuda")
    memory = estimate_memory(config, estimate_compute(config), cpu_only_hardware)
    assert memory.target_device == "cpu"
    assert memory.fits_gpu is None
    assert any("CUDA was requested" in note for note in memory.notes)


def test_random_forest_memory_has_no_gpu_path(cpu_only_hardware) -> None:
    config = _config("random_forest")
    memory = estimate_memory(config, estimate_compute(config), cpu_only_hardware)
    assert memory.target_device == "cpu"
    assert memory.gpu_training_vram_mb == 0
    assert memory.gpu_inference_vram_mb == 0
    assert memory.gpu_breakdown == []
    assert memory.fits_gpu is None


def test_precision_resolution_is_honest_on_cpu_and_gpu(
    cpu_only_hardware, gpu_hardware, pre_ampere_gpu_hardware
) -> None:
    fp16_cpu = resolve_precision(_config("mlp", precision="fp16"), cpu_only_hardware, "cpu")
    assert fp16_cpu.effective == "fp32"
    assert fp16_cpu.notes

    bf16_cpu = resolve_precision(_config("mlp", precision="bf16"), cpu_only_hardware, "cpu")
    assert bf16_cpu.effective == "bf16"

    fp16_gpu = resolve_precision(_config("mlp", precision="fp16"), gpu_hardware, "cuda")
    assert fp16_gpu.effective == "fp16"

    bf16_new = resolve_precision(_config("mlp", precision="bf16"), gpu_hardware, "cuda")
    assert bf16_new.effective == "bf16"

    bf16_old = resolve_precision(
        _config("mlp", precision="bf16"), pre_ampere_gpu_hardware, "cuda"
    )
    assert bf16_old.effective == "fp32"
    assert bf16_old.notes


# ---------------------------------------------------------------------------
# Time prediction
# ---------------------------------------------------------------------------


def test_time_prediction_range_is_ordered_and_positive(cpu_only_hardware) -> None:
    config = _config("mlp")
    compute = estimate_compute(config)
    estimate = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    assert 0 < estimate.predicted_low_seconds <= estimate.predicted_seconds
    assert estimate.predicted_seconds <= estimate.predicted_high_seconds
    assert estimate.predicted_hours >= 0
    assert estimate.throughput_samples_per_second > 0
    assert estimate.compute_seconds >= 0
    assert estimate.overhead_seconds >= 0
    assert estimate.precision_used == "fp32"
    assert "CPU" in estimate.bottleneck_primary or "batch" in estimate.bottleneck_primary


def test_more_data_increases_predicted_time(cpu_only_hardware) -> None:
    config = _config("mlp", dataset_size=1000)
    doubled = config.model_copy(update={"dataset_size": 2000})
    base = predict_training_time(config, estimate_compute(config), cpu_only_hardware, "cpu")
    larger = predict_training_time(doubled, estimate_compute(doubled), cpu_only_hardware, "cpu")
    assert larger.compute_seconds > 1.9 * base.compute_seconds
    assert larger.predicted_seconds > base.predicted_seconds


def test_larger_batch_reduces_predicted_time(cpu_only_hardware) -> None:
    small = _config("mlp", batch_size=8)
    large = small.model_copy(update={"batch_size": 128})
    small_time = predict_training_time(small, estimate_compute(small), cpu_only_hardware, "cpu")
    large_time = predict_training_time(large, estimate_compute(large), cpu_only_hardware, "cpu")
    assert large_time.predicted_seconds < small_time.predicted_seconds


def test_gpu_prediction_is_faster_than_cpu_for_the_same_model(
    cpu_only_hardware, gpu_hardware
) -> None:
    config = _config("mlp")
    compute = estimate_compute(config)
    cpu_time = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    gpu_time = predict_training_time(config, compute, gpu_hardware, "cuda")
    assert gpu_time.target_device == "cuda"
    assert gpu_time.effective_tflops > cpu_time.effective_tflops
    assert gpu_time.compute_seconds < cpu_time.compute_seconds

    # For a tiny workload, GPU launch/setup overhead can dominate the point
    # prediction. A compute-heavy model at a larger batch clearly benefits.
    heavy = _config(
        "cnn", dataset_size=50_000, batch_size=128, image_size=128, base_channels=32
    )
    heavy_compute = estimate_compute(heavy)
    cpu_heavy = predict_training_time(heavy, heavy_compute, cpu_only_hardware, "cpu")
    gpu_heavy = predict_training_time(heavy, heavy_compute, gpu_hardware, "cuda")
    assert gpu_heavy.predicted_seconds < cpu_heavy.predicted_seconds


def test_random_forest_time_ignores_epochs(cpu_only_hardware) -> None:
    config = _config("random_forest", epochs=1)
    many_epochs = config.model_copy(update={"epochs": 20})
    base = predict_training_time(config, estimate_compute(config), cpu_only_hardware, "cpu")
    more = predict_training_time(
        many_epochs, estimate_compute(many_epochs), cpu_only_hardware, "cpu"
    )
    assert base.target_device == "cpu"
    assert base.confidence == "low"
    assert base.predicted_seconds == more.predicted_seconds


def test_startup_can_be_excluded_for_benchmark_comparison(cpu_only_hardware) -> None:
    config = _config("mlp")
    compute = estimate_compute(config)
    full = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    no_startup = predict_training_time(
        config, compute, cpu_only_hardware, "cpu", include_startup=False
    )
    assert no_startup.predicted_seconds < full.predicted_seconds
    predicted, low, high, time_estimate, benchmark_compute = predict_benchmark_duration(
        config, cpu_only_hardware, "cpu", "fp32"
    )
    assert predicted == time_estimate.predicted_seconds
    assert low <= predicted <= high
    assert benchmark_compute.parameter_count == compute.parameter_count


# ---------------------------------------------------------------------------
# Suitability, bottlenecks, recommendations
# ---------------------------------------------------------------------------


def test_cpu_recommendations_are_bounded(cpu_only_hardware) -> None:
    config = _config("mlp")
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, cpu_only_hardware)
    time_estimate = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    recommendations = build_recommendations(
        config, compute, memory, time_estimate, cpu_only_hardware
    )
    suitability = build_suitability(
        config, compute, memory, time_estimate, cpu_only_hardware
    )

    assert recommendations.recommended_device == "cpu"
    assert 1 <= recommendations.cpu_threads <= cpu_only_hardware.cpu.logical_threads
    assert recommendations.dataloader_workers >= 1
    assert 1 <= recommendations.batch_size <= recommendations.max_safe_batch_size
    assert recommendations.precision == "fp32"
    assert recommendations.mixed_precision_available is False
    assert 0 <= suitability.cpu_score <= 100
    assert suitability.gpu_score == 0
    assert suitability.gpu_fit is None
    assert suitability.summary
    assert any("No NVIDIA GPU" in reason for reason in suitability.reasons)


def test_gpu_recommendations_offer_bf16_on_ampere(gpu_hardware) -> None:
    config = _config("mlp", batch_size=64)
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, gpu_hardware)
    time_estimate = predict_training_time(config, compute, gpu_hardware, "cuda")
    recommendations = build_recommendations(
        config, compute, memory, time_estimate, gpu_hardware
    )
    assert recommendations.recommended_device == "cuda"
    assert recommendations.mixed_precision_available is True
    assert recommendations.precision == "bf16"
    assert any("pin_memory" in advice for advice in recommendations.general_advice)
    assert mixed_precision_support(gpu_hardware) == (True, True)


def test_random_forest_recommendations_are_cpu_only(cpu_only_hardware) -> None:
    config = _config("random_forest")
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, cpu_only_hardware)
    time_estimate = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    recommendations = build_recommendations(
        config, compute, memory, time_estimate, cpu_only_hardware
    )
    assert recommendations.recommended_device == "cpu"
    assert recommendations.batch_size <= config.dataset_size
    assert recommendations.batch_size == recommendations.max_safe_batch_size
    assert any("max_depth" in advice for advice in recommendations.general_advice)


def test_bottlenecks_are_sorted_and_flag_deep_model_without_gpu(cpu_only_hardware) -> None:
    config = _config("transformer", sequence_length=1024)
    compute = estimate_compute(config)
    memory = estimate_memory(config, compute, cpu_only_hardware)
    time_estimate = predict_training_time(config, compute, cpu_only_hardware, "cpu")
    bottlenecks = build_bottlenecks(
        config, compute, memory, time_estimate, cpu_only_hardware
    )
    severities = [item.severity for item in bottlenecks]
    assert severities == sorted(
        severities, key=lambda value: {"high": 0, "medium": 1, "low": 2}[value]
    )
    assert any(
        item.category == "device" and item.severity == "high" for item in bottlenecks
    )
    assert any(
        item.category == "memory" and "sequence_length" in item.message
        for item in bottlenecks
    )


# ---------------------------------------------------------------------------
# Full planner response
# ---------------------------------------------------------------------------


def test_full_estimate_is_self_consistent_and_labelled(cpu_only_hardware) -> None:
    config = _config("mlp", dataset_size=512, epochs=2, batch_size=64)
    response = estimate_plan(config, hardware=cpu_only_hardware)

    assert response.config == config
    assert response.compute.parameter_count > 0
    assert response.memory.cpu_training_ram_mb > 0
    assert response.time.predicted_seconds > 0
    assert response.recommendations.recommended_device == "cpu"
    assert response.suitability.recommended_device == "cpu"
    assert response.bottlenecks
    assert response.disclaimer
    assert len(response.assumptions) >= 5
    assert response.estimation_seconds >= 0


def test_precision_and_cuda_fallbacks_create_warnings(cpu_only_hardware) -> None:
    config = _config("mlp", precision="fp16", device_preference="cuda")
    response = estimate_plan(config, hardware=cpu_only_hardware)
    assert response.time.precision_used == "fp32"
    assert any("effective precision is fp32" in warning for warning in response.warnings)
    assert any("CUDA was requested" in warning for warning in response.warnings)


def test_oversized_configuration_reports_memory_bottleneck(cpu_only_hardware) -> None:
    config = _config(
        "cnn",
        dataset_size=1000,
        batch_size=256,
        image_size=512,
        in_channels=3,
        base_channels=512,
        hidden_size=4096,
    )
    response = estimate_plan(config, hardware=cpu_only_hardware)
    assert response.memory.fits_cpu is False
    assert response.memory.fits_target is False
    assert response.bottlenecks[0].category == "memory"
    assert response.bottlenecks[0].severity == "high"
    assert any("does not fit" in warning for warning in response.warnings)
