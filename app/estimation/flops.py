"""Analytical parameter-count and FLOP estimators.

Conventions:
* One multiply-add is counted as 2 FLOPs (one multiply, one add).
* Elementwise activations, normalization, pooling, and optimizer math are excluded
  from dense/conv/attention FLOP totals unless stated otherwise.
* Training FLOPs are approximated as 3x forward FLOPs (forward + backward), a
  standard planning heuristic. Actual ratio depends on the model and framework.
* Random Forest cost is reported as an approximate comparison/operation count
  because trees do not have a meaningful dense-FLOP equivalent.
"""

from __future__ import annotations

import math

from app.core.config import TRAINING_BACKWARD_MULTIPLIER
from app.models.architectures import CNN_FLATTEN_PER_CHANNEL
from app.schemas import ComputeEstimate, LayerBreakdown, ModelConfig


def _mlp_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    layers: list[LayerBreakdown] = []
    previous = config.features
    for index in range(config.num_layers):
        params = previous * config.hidden_size + config.hidden_size
        layers.append(
            LayerBreakdown(
                name=f"Dense {index + 1} ({previous} → {config.hidden_size})",
                parameters=params,
                forward_flops=2 * previous * config.hidden_size,
                activation_elements=config.hidden_size,
            )
        )
        previous = config.hidden_size
    params = previous * config.num_classes + config.num_classes
    layers.append(
        LayerBreakdown(
            name=f"Output ({previous} → {config.num_classes})",
            parameters=params,
            forward_flops=2 * previous * config.num_classes,
            activation_elements=config.num_classes,
        )
    )
    return layers


def _cnn_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    c1 = config.base_channels
    c2 = c1 * 2
    c3 = c1 * 4
    h = config.image_size
    h1 = max(1, h // 2)
    h2 = max(1, h1 // 2)
    flatten = c3 * CNN_FLATTEN_PER_CHANNEL
    layers = [
        LayerBreakdown(
            name=f"Conv2d 1 ({config.in_channels} → {c1}, 3×3)",
            parameters=config.in_channels * c1 * 9 + c1,
            forward_flops=2 * h * h * config.in_channels * c1 * 9,
            activation_elements=c1 * h1 * h1,
        ),
        LayerBreakdown(
            name=f"Conv2d 2 ({c1} → {c2}, 3×3)",
            parameters=c1 * c2 * 9 + c2,
            forward_flops=2 * h1 * h1 * c1 * c2 * 9,
            activation_elements=c2 * h2 * h2,
        ),
        LayerBreakdown(
            name=f"Conv2d 3 ({c2} → {c3}, 3×3) + adaptive pool",
            parameters=c2 * c3 * 9 + c3,
            forward_flops=2 * h2 * h2 * c2 * c3 * 9,
            activation_elements=c3 * CNN_FLATTEN_PER_CHANNEL,
        ),
        LayerBreakdown(
            name=f"Dense ({flatten} → {config.hidden_size})",
            parameters=flatten * config.hidden_size + config.hidden_size,
            forward_flops=2 * flatten * config.hidden_size,
            activation_elements=config.hidden_size,
        ),
        LayerBreakdown(
            name=f"Output ({config.hidden_size} → {config.num_classes})",
            parameters=config.hidden_size * config.num_classes + config.num_classes,
            forward_flops=2 * config.hidden_size * config.num_classes,
            activation_elements=config.num_classes,
        ),
    ]
    return layers


def _lstm_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    h = config.hidden_size
    total_params = 0
    for layer_index in range(config.num_layers):
        input_dim = config.features if layer_index == 0 else h
        layer_params = 4 * h * input_dim + 4 * h * h + 8 * h
        total_params += layer_params
    head_params = h * config.num_classes + config.num_classes
    forward_flops = 0
    for layer_index in range(config.num_layers):
        input_dim = config.features if layer_index == 0 else h
        forward_flops += (
            config.sequence_length * 2 * 4 * h * (input_dim + h)
        )
    forward_flops += 2 * h * config.num_classes
    activation_elements = (
        config.sequence_length * h * config.num_layers * 4
        + config.sequence_length * h * 2
    )
    return [
        LayerBreakdown(
            name=(
                f"LSTM ({config.features} → {h}, {config.num_layers} layers, "
                f"{config.sequence_length} steps)"
            ),
            parameters=total_params,
            forward_flops=forward_flops - 2 * h * config.num_classes,
            activation_elements=activation_elements,
        ),
        LayerBreakdown(
            name=f"Classifier ({h} → {config.num_classes})",
            parameters=head_params,
            forward_flops=2 * h * config.num_classes,
            activation_elements=config.num_classes,
        ),
    ]


def _transformer_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    d = config.hidden_size
    f = config.features
    seq = config.sequence_length
    heads = config.num_heads
    layers_count = config.num_layers

    per_layer_params = 12 * d * d + 13 * d
    per_layer_flops = 24 * seq * d * d + 4 * seq * seq * d
    per_layer_activations = 10 * seq * d + heads * seq * seq

    return [
        LayerBreakdown(
            name=f"Input projection ({f} → {d})",
            parameters=f * d + d,
            forward_flops=2 * seq * f * d,
            activation_elements=seq * d,
        ),
        LayerBreakdown(
            name=f"Positional embedding ({seq} × {d})",
            parameters=seq * d,
            forward_flops=0,
            activation_elements=0,
        ),
        LayerBreakdown(
            name=f"Transformer encoder × {layers_count}",
            parameters=layers_count * per_layer_params,
            forward_flops=layers_count * per_layer_flops,
            activation_elements=layers_count * per_layer_activations,
        ),
        LayerBreakdown(
            name=f"Classifier ({d} → {config.num_classes})",
            parameters=d * config.num_classes + config.num_classes,
            forward_flops=2 * d * config.num_classes,
            activation_elements=config.num_classes,
        ),
    ]


def _rf_effective_depth(config: ModelConfig) -> int:
    sample_bound = max(1, math.ceil(math.log2(max(2, config.dataset_size))))
    return max(1, min(config.max_depth, sample_bound))


def _rf_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    depth = _rf_effective_depth(config)
    max_nodes = min(2 ** (depth + 1) - 1, 2 * config.dataset_size - 1)
    scalars_per_tree = max_nodes * 3  # feature/threshold/leaf value approximation
    params = config.n_estimators * scalars_per_tree
    features_tried = max(1, round(math.sqrt(config.features)))
    forward_flops_per_sample = config.n_estimators * depth * 2
    # A coarse work estimate for split search and histogram construction.
    training_ops = (
        config.n_estimators
        * config.dataset_size
        * depth
        * features_tried
        * 4
    )
    return [
        LayerBreakdown(
            name=(
                f"Random forest ({config.n_estimators} trees, depth ≤ {depth}, "
                f"≈{max_nodes} nodes/tree)"
            ),
            parameters=params,
            forward_flops=forward_flops_per_sample,
            activation_elements=0,
        )
        if training_ops >= 0
        else LayerBreakdown(  # pragma: no cover - unreachable guard
            name="Random forest", parameters=0, forward_flops=0, activation_elements=0
        )
    ]


def _linear_breakdown(config: ModelConfig) -> list[LayerBreakdown]:
    return [
        LayerBreakdown(
            name=f"Linear ({config.features} → 1)",
            parameters=config.features + 1,
            forward_flops=2 * config.features,
            activation_elements=1,
        )
    ]


def estimate_compute(config: ModelConfig) -> ComputeEstimate:
    """Estimate parameters, forward FLOPs, and full training workload."""
    notes: list[str] = []
    if config.model_type == "random_forest":
        breakdown = _rf_breakdown(config)
        depth = _rf_effective_depth(config)
        features_tried = max(1, round(math.sqrt(config.features)))
        forward_per_sample = config.n_estimators * depth * 2
        training_total = (
            config.n_estimators
            * config.dataset_size
            * depth
            * features_tried
            * 4
        )
        params = sum(item.parameters for item in breakdown)
        notes.extend(
            [
                "Random Forest has no gradient-based backward pass; training cost is "
                "an approximate comparison/histogram operation count.",
                "The configured epochs value is ignored for Random Forest.",
                f"Effective tree depth is limited to {depth} by dataset size and max_depth.",
            ]
        )
        return ComputeEstimate(
            parameter_count=params,
            trainable_parameters=params,
            forward_flops_per_sample=forward_per_sample,
            backward_flops_per_sample=0,
            training_flops_per_sample=training_total // config.dataset_size,
            total_forward_flops=forward_per_sample * config.dataset_size,
            total_training_flops=training_total,
            optimizer_steps=0,
            activation_elements=0,
            breakdown=breakdown,
            notes=notes,
        )

    builders = {
        "linear_regression": _linear_breakdown,
        "mlp": _mlp_breakdown,
        "cnn": _cnn_breakdown,
        "lstm": _lstm_breakdown,
        "transformer": _transformer_breakdown,
    }
    breakdown = builders[config.model_type](config)
    params = sum(item.parameters for item in breakdown)
    forward = sum(item.forward_flops for item in breakdown)
    training_per_sample = int(round(forward * (1.0 + TRAINING_BACKWARD_MULTIPLIER)))
    total_training = training_per_sample * config.dataset_size * config.epochs
    optimizer_steps = math.ceil(config.dataset_size / config.batch_size) * config.epochs
    notes.append(
        "Training FLOPs ≈ 3 × forward FLOPs per sample; optimizer state-update math "
        "and data loading are excluded."
    )
    if config.model_type == "transformer":
        notes.append(
            "Attention FLOPs scale with sequence_length²; activation memory does as well."
        )
    return ComputeEstimate(
        parameter_count=params,
        trainable_parameters=params,
        forward_flops_per_sample=forward,
        backward_flops_per_sample=int(round(forward * TRAINING_BACKWARD_MULTIPLIER)),
        training_flops_per_sample=training_per_sample,
        total_forward_flops=forward * config.dataset_size * config.epochs,
        total_training_flops=total_training,
        optimizer_steps=optimizer_steps,
        activation_elements=sum(item.activation_elements for item in breakdown),
        breakdown=breakdown,
        notes=notes,
    )
