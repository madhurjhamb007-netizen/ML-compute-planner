"""Actual (small) benchmark workloads.

All workloads are deliberately bounded by the runner. They use synthetic data and
reuse one pre-generated batch so the measurement reflects model compute and the
optimizer step rather than disk or DataLoader variance. This limitation is
reported with every benchmark result.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.models.architectures import build_synthetic_batch, build_torch_model
from app.schemas import ModelConfig, Precision, ResolvedDevice


@dataclass
class WorkloadStats:
    steps_completed: int
    samples_seen: int
    duration_seconds: float
    cpu_time_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


class TorchWorkload:
    def __init__(
        self,
        config: ModelConfig,
        device: ResolvedDevice,
        precision: Precision,
        batch_size: int,
        steps: int,
        warmup_steps: int,
        max_seconds: float,
    ) -> None:
        self.config = config
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.max_seconds = max_seconds
        self.model: Any = None
        self.optimizer: Any = None
        self.criterion: Any = None
        self.x: Any = None
        self.y: Any = None
        self.torch: Any = None
        self.scaler: Any = None
        self.autocast_dtype: Any = None
        self.warnings: list[str] = []

    def prepare(self) -> None:
        import torch

        self.torch = torch
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = build_torch_model(self.config).to(self.device)
        if self.config.model_type == "linear_regression":
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.01, momentum=0.9
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.x, self.y = build_synthetic_batch(
            self.config, self.batch_size, self.device
        )
        self.autocast_dtype = None
        self.scaler = None
        self._setup_amp()

        # Warmup is excluded from measured duration.
        warmup_deadline = time.perf_counter() + max(2.0, self.max_seconds * 0.35)
        completed = 0
        for _ in range(self.warmup_steps):
            self._step()
            completed += 1
            if time.perf_counter() > warmup_deadline:
                self.warnings.append(
                    "Warmup was stopped early to respect the benchmark time budget."
                )
                break
        if completed < self.warmup_steps:
            self.warnings.append(
                f"Only {completed}/{self.warmup_steps} warmup steps completed."
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _setup_amp(self) -> None:
        torch = self.torch
        if self.precision == "fp16":
            self.autocast_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        elif self.precision == "bf16":
            self.autocast_dtype = torch.bfloat16
            self.scaler = None

    def _step(self) -> None:
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        device_type = "cuda" if self.device == "cuda" else "cpu"
        enabled = self.autocast_dtype is not None
        with torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=enabled,
        ):
            output = self.model(self.x)
            loss = self.criterion(output, self.y)
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def measure(self) -> WorkloadStats:
        assert self.torch is not None, "prepare() must be called first"
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        completed = 0
        started = time.perf_counter()
        cpu_started = time.process_time()
        for _ in range(self.steps):
            self._step()
            completed += 1
            if self.device == "cuda":
                torch.cuda.synchronize()
            if time.perf_counter() - started >= self.max_seconds:
                if completed < self.steps:
                    self.warnings.append(
                        f"Stopped after {completed}/{self.steps} steps to respect the "
                        f"{self.max_seconds:.0f}s benchmark safety limit."
                    )
                break
        if self.device == "cuda":
            torch.cuda.synchronize()
        duration = max(0.0, time.perf_counter() - started)
        cpu_time = max(0.0, time.process_time() - cpu_started)
        return WorkloadStats(
            steps_completed=completed,
            samples_seen=completed * self.batch_size,
            duration_seconds=round(duration, 6),
            cpu_time_seconds=round(cpu_time, 6),
            warnings=self.warnings,
        )

    def cleanup(self) -> None:
        try:
            del self.model, self.optimizer, self.x, self.y
        except AttributeError:
            pass
        if self.torch is not None and self.device == "cuda":
            self.torch.cuda.empty_cache()


class MiniRandomForest:
    """Small histogram-style random forest classifier implemented with NumPy.

    It performs real recursive split search and bootstrap sampling. It exists to
    benchmark tree-training cost without adding scikit-learn as a dependency.
    """

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        n_features: int,
        n_classes: int,
        max_features: int | None = None,
        n_bins: int = 6,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_features = max_features or max(1, int(math.sqrt(n_features)))
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(random_state)
        self.nodes_total = 0
        self.trees_fitted = 0
        self.early_stop = False

    def _gini(self, counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - np.sum(probabilities**2))

    def _build_tree(
        self, x: np.ndarray, y: np.ndarray, deadline: float | None
    ) -> int:
        n_samples = x.shape[0]
        stack: list[tuple[np.ndarray, int]] = [(np.arange(n_samples), 0)]
        nodes = 0
        while stack:
            indices, depth = stack.pop()
            nodes += 1
            node_y = y[indices]
            if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
                continue
            counts = np.bincount(node_y, minlength=self.n_classes)
            if np.count_nonzero(counts) <= 1:
                continue

            feature_ids = self.rng.choice(
                self.n_features, size=min(self.max_features, self.n_features), replace=False
            )
            best_score = float("inf")
            best_feature: int | None = None
            best_threshold = 0.0
            n_total = len(indices)
            for feature in feature_ids:
                values = x[indices, feature]
                quantiles = np.quantile(
                    values, np.linspace(0.15, 0.85, self.n_bins - 1)
                )
                for threshold in np.unique(quantiles):
                    mask = values <= threshold
                    n_left = int(mask.sum())
                    n_right = n_total - n_left
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    left_counts = np.bincount(node_y[mask], minlength=self.n_classes)
                    right_counts = counts - left_counts
                    score = (
                        n_left / n_total * self._gini(left_counts)
                        + n_right / n_total * self._gini(right_counts)
                    )
                    if score < best_score:
                        best_score = score
                        best_feature = int(feature)
                        best_threshold = float(threshold)
            if best_feature is None:
                continue
            mask = x[indices, best_feature] <= best_threshold
            stack.append((indices[mask], depth + 1))
            stack.append((indices[~mask], depth + 1))
            if deadline is not None and nodes % 64 == 0 and time.perf_counter() > deadline:
                self.early_stop = True
                break
        return nodes

    def fit(self, x: np.ndarray, y: np.ndarray, max_seconds: float) -> None:
        deadline = time.perf_counter() + max_seconds
        for _ in range(self.n_estimators):
            bootstrap = self.rng.integers(0, x.shape[0], size=x.shape[0])
            self.nodes_total += self._build_tree(x[bootstrap], y[bootstrap], deadline)
            self.trees_fitted += 1
            if self.early_stop or time.perf_counter() > deadline:
                self.early_stop = True
                break


class RandomForestWorkload:
    def __init__(
        self,
        config: ModelConfig,
        samples: int,
        max_seconds: float,
        n_jobs: int = 1,
    ) -> None:
        self.config = config
        self.samples = samples
        self.max_seconds = max_seconds
        self.n_jobs = max(1, n_jobs)
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.warnings: list[str] = []
        self.forest: MiniRandomForest | None = None
        self.backend = "numpy"
        self._sklearn_rf = None
        try:
            from sklearn.ensemble import RandomForestClassifier

            self._sklearn_rf = RandomForestClassifier
            self.backend = "scikit-learn"
        except Exception:  # noqa: BLE001 - optional acceleration
            self.warnings.append(
                "scikit-learn is unavailable; using the bundled NumPy forest benchmark. "
                "Its runtime is not representative of optimized tree libraries."
            )

    def prepare(self) -> None:
        rng = np.random.default_rng(42)
        self.x = rng.standard_normal(
            (self.samples, self.config.features), dtype=np.float32
        )
        self.y = rng.integers(0, self.config.num_classes, self.samples, dtype=np.int64)
        self.warnings.append(
            f"Random Forest benchmark backend: {self.backend} with {self.n_jobs} jobs."
        )

    def measure(self) -> WorkloadStats:
        assert self.x is not None and self.y is not None
        if self._sklearn_rf is not None:
            model = self._sklearn_rf(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=self.n_jobs,
                random_state=42,
            )
            started = time.perf_counter()
            cpu_started = time.process_time()
            model.fit(self.x, self.y)
            duration = max(0.0, time.perf_counter() - started)
            cpu_time = max(0.0, time.process_time() - cpu_started)
            nodes = int(sum(tree.tree_.node_count for tree in model.estimators_))
            self.warnings.append(
                f"scikit-learn forest fitted {len(model.estimators_)} trees with "
                f"{nodes} total nodes."
            )
            if duration > self.max_seconds:
                self.warnings.append(
                    "The Random Forest fit exceeded the requested time budget; the "
                    "workload was already bounded by configuration limits."
                )
            return WorkloadStats(
                steps_completed=1,
                samples_seen=self.samples,
                duration_seconds=round(duration, 6),
                cpu_time_seconds=round(cpu_time, 6),
                warnings=self.warnings,
            )

        self.forest = MiniRandomForest(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            n_features=self.config.features,
            n_classes=self.config.num_classes,
        )
        started = time.perf_counter()
        cpu_started = time.process_time()
        self.forest.fit(self.x, self.y, max_seconds=self.max_seconds)
        duration = max(0.0, time.perf_counter() - started)
        cpu_time = max(0.0, time.process_time() - cpu_started)
        if self.forest.early_stop or self.forest.trees_fitted < self.config.n_estimators:
            self.warnings.append(
                f"Random Forest stopped after {self.forest.trees_fitted}/"
                f"{self.config.n_estimators} trees due to the time budget."
            )
        self.warnings.append(
            f"Forest fitted {self.forest.trees_fitted} trees with "
            f"{self.forest.nodes_total} total nodes."
        )
        return WorkloadStats(
            steps_completed=1,
            samples_seen=self.samples,
            duration_seconds=round(duration, 6),
            cpu_time_seconds=round(cpu_time, 6),
            warnings=self.warnings,
        )

    def cleanup(self) -> None:
        self.x = None
        self.y = None
        self.forest = None
