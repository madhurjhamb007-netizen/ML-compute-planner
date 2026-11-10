"""Real PyTorch model factories used by benchmark workloads.

The analytical estimators in ``app.estimation.flops`` intentionally mirror these
architectures. Tests compare the analytical parameter counts against PyTorch's
``sum(p.numel())`` to keep the two implementations aligned.
"""

from __future__ import annotations

from typing import Any

from app.schemas import ModelConfig

CNN_FLATTEN_PER_CHANNEL = 16  # AdaptiveAvgPool2d(4) -> 4 * 4 spatial positions


def _require_torch():  # type: ignore[no-untyped-def]
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required for benchmarking. Install it with "
            "`pip install torch`."
        ) from exc
    return torch, nn


def _build_cnn(config: ModelConfig, nn: Any):  # type: ignore[no-untyped-def]
    c1 = config.base_channels
    c2 = c1 * 2
    c3 = c1 * 4
    flatten = c3 * CNN_FLATTEN_PER_CHANNEL

    class CNNClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(config.in_channels, c1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(flatten, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.num_classes),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.classifier(self.features(x))

    return CNNClassifier()


def _build_lstm(config: ModelConfig, nn: Any):  # type: ignore[no-untyped-def]
    class LSTMClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=config.features,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(config.hidden_size, config.num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            _, (hidden, _) = self.encoder(x)
            return self.head(hidden[-1])

    return LSTMClassifier()


def _build_transformer(config: ModelConfig, torch: Any, nn: Any):  # type: ignore[no-untyped-def]
    class TransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(config.features, config.hidden_size)
            self.positional = nn.Parameter(
                torch.zeros(config.sequence_length, config.hidden_size)
            )
            nn.init.normal_(self.positional, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=4 * config.hidden_size,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.head = nn.Linear(config.hidden_size, config.num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.input_projection(x) + self.positional.unsqueeze(0)
            x = self.encoder(x)
            return self.head(x.mean(dim=1))

    return TransformerClassifier()


def build_torch_model(config: ModelConfig) -> Any:
    """Create the benchmark model for a configuration.

    Random Forest is not a PyTorch model and is handled by the NumPy workload.
    """
    torch, nn = _require_torch()
    if config.model_type == "linear_regression":
        return nn.Linear(config.features, 1)
    if config.model_type == "mlp":
        modules: list[Any] = [nn.Linear(config.features, config.hidden_size), nn.ReLU()]
        for _ in range(config.num_layers - 1):
            modules.extend([nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU()])
        modules.append(nn.Linear(config.hidden_size, config.num_classes))
        return nn.Sequential(*modules)
    if config.model_type == "cnn":
        return _build_cnn(config, nn)
    if config.model_type == "lstm":
        return _build_lstm(config, nn)
    if config.model_type == "transformer":
        return _build_transformer(config, torch, nn)
    raise ValueError(f"No PyTorch architecture for model type: {config.model_type}")


def build_synthetic_batch(config: ModelConfig, batch_size: int, device: str):
    """Create a synthetic input/label pair with the right shape for the model."""
    torch, _ = _require_torch()

    if config.model_type == "cnn":
        x = torch.randn(batch_size, config.in_channels, config.image_size, config.image_size)
    elif config.model_type in ("lstm", "transformer"):
        x = torch.randn(batch_size, config.sequence_length, config.features)
    else:
        x = torch.randn(batch_size, config.features)
    x = x.to(device)

    if config.model_type == "linear_regression":
        y = torch.randn(batch_size, 1, device=device)
    else:
        y = torch.randint(0, config.num_classes, (batch_size,), device=device)
    return x, y
