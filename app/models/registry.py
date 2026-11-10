"""Model metadata and default configuration."""

from __future__ import annotations

from app.schemas import ModelConfig, ModelInfo, ModelType

_MODEL_ORDER: tuple[ModelType, ...] = (
    "linear_regression",
    "random_forest",
    "mlp",
    "cnn",
    "lstm",
    "transformer",
)

_SPECS: dict[ModelType, dict[str, object]] = {
    "linear_regression": {
        "display_name": "Linear Regression",
        "family": "Classical / linear",
        "description": (
            "Single dense layer trained with mean-squared error. Useful as a "
            "low-cost baseline for tabular regression."
        ),
        "supports_gpu": True,
        "gpu_friendly": False,
        "notes": "Parameter and FLOP formulas are exact for a single Linear layer.",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "family": "Classical / tree ensemble",
        "description": (
            "CPU tree ensemble benchmark implemented with NumPy. Parameters, FLOPs, "
            "and training cost are estimates based on tree structure."
        ),
        "supports_gpu": False,
        "gpu_friendly": False,
        "notes": (
            "Tree training is CPU-bound and not accelerated by the PyTorch GPU path. "
            "Benchmarks use scikit-learn when available with a NumPy fallback. "
            "Epochs and precision do not apply."
        ),
    },
    "mlp": {
        "display_name": "Multi-Layer Perceptron",
        "family": "Feed-forward neural network",
        "description": (
            "Fully connected network for tabular or vector data with configurable "
            "width and depth."
        ),
        "supports_gpu": True,
        "gpu_friendly": True,
        "notes": "Parameter and dense FLOP formulas are exact for the implemented MLP.",
    },
    "cnn": {
        "display_name": "Convolutional Neural Network",
        "family": "Convolutional neural network",
        "description": (
            "Three convolution blocks with batch pooling plus a dense classifier. "
            "Image size and channel count are configurable."
        ),
        "supports_gpu": True,
        "gpu_friendly": True,
        "notes": "Conv FLOPs count multiply-add operations; pooling is ignored.",
    },
    "lstm": {
        "display_name": "LSTM",
        "family": "Recurrent neural network",
        "description": (
            "Stacked LSTM encoder over a synthetic sequence, followed by a linear "
            "classification head."
        ),
        "supports_gpu": True,
        "gpu_friendly": True,
        "notes": "Actual PyTorch LSTM speed can differ substantially from FLOP estimates.",
    },
    "transformer": {
        "display_name": "Transformer Encoder",
        "family": "Attention-based neural network",
        "description": (
            "Pre-embedding projection, learned positional embeddings, a stack of "
            "Transformer encoder layers, mean pooling, and a classification head."
        ),
        "supports_gpu": True,
        "gpu_friendly": True,
        "notes": "Attention memory scales quadratically with sequence length.",
    },
}


def default_config(model_type: ModelType) -> ModelConfig:
    base = ModelConfig(model_type=model_type)
    updates: dict[str, object] = {}
    if model_type == "linear_regression":
        updates = {"features": 32, "epochs": 20, "batch_size": 64}
    elif model_type == "random_forest":
        updates = {
            "features": 32,
            "epochs": 1,
            "n_estimators": 100,
            "max_depth": 10,
            "num_classes": 2,
            "batch_size": 256,
        }
    elif model_type == "mlp":
        updates = {
            "features": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "epochs": 10,
        }
    elif model_type == "cnn":
        updates = {
            "in_channels": 3,
            "image_size": 32,
            "base_channels": 32,
            "hidden_size": 128,
            "epochs": 10,
        }
    elif model_type == "lstm":
        updates = {
            "features": 32,
            "sequence_length": 64,
            "hidden_size": 128,
            "num_layers": 2,
            "epochs": 10,
        }
    elif model_type == "transformer":
        updates = {
            "features": 32,
            "sequence_length": 64,
            "hidden_size": 128,
            "num_heads": 4,
            "num_layers": 2,
            "epochs": 10,
        }
    return base.model_copy(update=updates)


def get_models() -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for key in _MODEL_ORDER:
        spec = _SPECS[key]
        models.append(
            ModelInfo(
                key=key,
                display_name=str(spec["display_name"]),
                family=str(spec["family"]),
                description=str(spec["description"]),
                supports_gpu=bool(spec["supports_gpu"]),
                gpu_friendly=bool(spec["gpu_friendly"]),
                default_config=default_config(key),
                notes=str(spec["notes"]),
            )
        )
    return models


def get_model_info(model_type: ModelType) -> ModelInfo:
    for model in get_models():
        if model.key == model_type:
            return model
    raise ValueError(f"Unknown model type: {model_type}")
