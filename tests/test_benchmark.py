"""Benchmark planning, safety caps, and real execution tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.benchmark.runner import (
    BenchmarkSafetyError,
    build_plan,
    run_benchmark,
)
from app.benchmark.workloads import MiniRandomForest
from app.schemas import BenchmarkOptions, BenchmarkRequest


def _request(model_type: str, **updates) -> BenchmarkRequest:
    benchmark = updates.pop("benchmark", None)
    payload = {"model_type": model_type, **updates}
    if benchmark is not None:
        payload["benchmark"] = benchmark
    return BenchmarkRequest(**payload)


def test_torch_benchmark_plan_enforces_all_safety_caps(cpu_only_hardware) -> None:
    request = _request(
        "mlp",
        dataset_size=10_000,
        features=16,
        hidden_size=32,
        num_layers=1,
        batch_size=4096,
        benchmark={
            "device": "cpu",
            "steps": 100,
            "warmup_steps": 5,
            "samples": 2048,
            "batch_size": 256,
            "max_seconds": 15,
        },
    )
    plan = build_plan(request, cpu_only_hardware)

    assert plan.mode == "torch"
    assert plan.device == "cpu"
    assert plan.samples <= 2048
    assert plan.batch_size <= 256
    assert plan.steps <= 100
    assert plan.warmup_steps <= 5
    assert plan.samples == plan.steps * plan.batch_size
    assert plan.effective_config.epochs == 1
    assert plan.effective_config.dataset_size == plan.samples
    assert plan.safety_notes


def test_random_forest_plan_enforces_tree_specific_caps(cpu_only_hardware) -> None:
    request = _request(
        "random_forest",
        dataset_size=5000,
        features=999,
        n_estimators=500,
        max_depth=40,
        num_classes=2,
        benchmark={
            "device": "cpu",
            "steps": 50,
            "warmup_steps": 5,
            "samples": 2048,
            "max_seconds": 15,
        },
    )
    plan = build_plan(request, cpu_only_hardware)

    assert plan.mode == "rf"
    assert plan.device == "cpu"
    assert plan.samples <= 1200
    assert plan.effective_config.features <= 256
    assert plan.effective_config.n_estimators <= 40
    assert plan.effective_config.max_depth <= 10
    assert plan.effective_config.precision == "fp32"
    assert plan.effective_config.epochs == 1
    assert plan.safety_notes


def test_cuda_request_falls_back_to_cpu_when_unavailable(cpu_only_hardware) -> None:
    plan = build_plan(
        _request(
            "mlp",
            dataset_size=16,
            features=4,
            hidden_size=8,
            num_layers=1,
            benchmark={"device": "cuda", "steps": 1, "warmup_steps": 0, "samples": 8},
        ),
        cpu_only_hardware,
    )
    assert plan.device == "cpu"
    assert any("CUDA was requested" in warning for warning in plan.warnings)


def test_benchmark_rejects_a_workload_that_does_not_fit(cpu_only_hardware) -> None:
    request = _request(
        "cnn",
        dataset_size=2048,
        batch_size=256,
        image_size=512,
        in_channels=3,
        base_channels=512,
        hidden_size=4096,
        benchmark={
            "device": "cpu",
            "steps": 1,
            "warmup_steps": 0,
            "samples": 2048,
            "max_seconds": 1,
        },
    )
    with pytest.raises(BenchmarkSafetyError):
        build_plan(request, cpu_only_hardware)


def test_tiny_mlp_benchmark_produces_real_measurements(cpu_only_hardware) -> None:
    request = _request(
        "mlp",
        dataset_size=16,
        features=4,
        hidden_size=8,
        num_layers=1,
        num_classes=2,
        batch_size=4,
        benchmark={
            "device": "cpu",
            "steps": 1,
            "warmup_steps": 0,
            "samples": 8,
            "max_seconds": 3,
        },
    )
    response = run_benchmark(request, hardware=cpu_only_hardware)

    result = response.result
    assert result.run_id
    assert result.model_type == "mlp"
    assert result.device_used == "cpu"
    assert result.steps_completed == 1
    assert result.samples_used == 4
    assert result.duration_seconds > 0
    assert result.samples_per_second > 0
    assert result.training_flops_performed > 0
    assert result.ram_peak_mb > 0
    assert result.samples, "short benchmark should still capture at least one sample"
    assert response.comparison.measured_seconds > 0
    assert response.comparison.predicted_seconds > 0
    assert response.comparison.time_ratio > 0
    assert response.effective_config.dataset_size == result.samples_used
    assert response.safety_notes
    assert response.disclaimer


def test_tiny_random_forest_benchmark_produces_real_measurements(cpu_only_hardware) -> None:
    request = _request(
        "random_forest",
        dataset_size=32,
        features=4,
        num_classes=2,
        n_estimators=2,
        max_depth=2,
        batch_size=8,
        benchmark={
            "device": "cpu",
            "steps": 1,
            "warmup_steps": 0,
            "samples": 32,
            "max_seconds": 5,
        },
    )
    response = run_benchmark(request, hardware=cpu_only_hardware)

    result = response.result
    assert result.model_type == "random_forest"
    assert result.device_used == "cpu"
    assert result.steps_completed == 1
    assert result.samples_used == 32
    assert result.duration_seconds > 0
    assert result.training_flops_performed > 0
    assert response.effective_config.dataset_size == 32
    assert any("backend" in warning.lower() for warning in result.warnings)
    assert "Random Forest" in response.comparison.interpretation


def test_numpy_forest_fallback_really_fits_trees() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 4), dtype=np.float32)
    y = rng.integers(0, 2, 64, dtype=np.int64)
    forest = MiniRandomForest(
        n_estimators=2,
        max_depth=2,
        n_features=4,
        n_classes=2,
        random_state=7,
    )
    forest.fit(x, y, max_seconds=5.0)
    assert forest.trees_fitted == 2
    assert forest.nodes_total >= 2
    assert forest.early_stop is False
