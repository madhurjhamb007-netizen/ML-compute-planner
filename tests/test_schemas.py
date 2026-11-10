"""Schema validation and model-registry tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.registry import default_config, get_model_info, get_models
from app.schemas import BenchmarkOptions, BenchmarkRequest, ModelConfig


EXPECTED_MODELS = [
    "linear_regression",
    "random_forest",
    "mlp",
    "cnn",
    "lstm",
    "transformer",
]


def test_registry_exposes_all_models_in_order() -> None:
    models = get_models()
    assert [model.key for model in models] == EXPECTED_MODELS
    for model in models:
        assert model.display_name
        assert model.description
        assert model.default_config.model_type == model.key


@pytest.mark.parametrize("model_type", EXPECTED_MODELS)
def test_every_default_config_is_valid(model_type: str) -> None:
    config = default_config(model_type)  # type: ignore[arg-type]
    assert config.model_type == model_type
    assert 1 <= config.batch_size <= 4096
    assert 1 <= config.dataset_size <= 10_000_000


def test_get_model_info_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        get_model_info("not_a_model")  # type: ignore[arg-type]


def test_transformer_requires_divisible_attention_heads() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(
            model_type="transformer",
            hidden_size=130,
            num_heads=4,
        )


def test_transformer_rejects_more_heads_than_hidden_units() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        ModelConfig(
            model_type="transformer",
            hidden_size=8,
            num_heads=16,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_size", 0),
        ("batch_size", 0),
        ("features", 0),
        ("epochs", 0),
        ("num_layers", 65),
        ("num_classes", 1),
        ("image_size", 8),
        ("base_channels", 2048),
    ],
)
def test_numeric_bounds_are_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(**{field: value})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"model_type": "mlp", "made_up_field": 1})


def test_benchmark_options_have_conservative_limits() -> None:
    with pytest.raises(ValidationError):
        BenchmarkOptions(steps=101)
    with pytest.raises(ValidationError):
        BenchmarkOptions(warmup_steps=6)
    with pytest.raises(ValidationError):
        BenchmarkOptions(samples=4096)
    with pytest.raises(ValidationError):
        BenchmarkOptions(batch_size=512)
    with pytest.raises(ValidationError):
        BenchmarkOptions(max_seconds=20)


def test_benchmark_request_inherits_config_and_nests_options() -> None:
    request = BenchmarkRequest(
        model_type="mlp",
        dataset_size=64,
        benchmark={"steps": 2, "samples": 32, "warmup_steps": 0},
    )
    assert request.benchmark.steps == 2
    assert request.dataset_size == 64


def test_random_forest_default_is_cpu_friendly() -> None:
    config = default_config("random_forest")
    assert config.epochs == 1
    assert config.n_estimators > 0
    assert config.max_depth > 0
