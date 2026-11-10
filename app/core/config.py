"""Application constants.

Memory values use MiB (1 MiB = 1024 * 1024 bytes) throughout the project,
even where the shorter label "MB" appears in the UI.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BYTES_PER_MIB = 1024**2
BYTES_PER_GIB = 1024**3

APP_NAME = "ML Compute Planner"
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Safety limits for benchmark execution. These are deliberately conservative.
BENCHMARK_MAX_STEPS = 100
BENCHMARK_MAX_WARMUP_STEPS = 5
BENCHMARK_MAX_SAMPLES = 2048
BENCHMARK_MAX_BATCH_SIZE = 256
BENCHMARK_MAX_SECONDS = 15.0
BENCHMARK_MAX_RF_SAMPLES = 1200
BENCHMARK_MAX_RF_TREES = 40
BENCHMARK_MAX_RF_DEPTH = 10
BENCHMARK_DEFAULT_SAMPLES = 512
BENCHMARK_DEFAULT_STEPS = 20
BENCHMARK_DEFAULT_WARMUP = 2

# Estimation assumptions (documented in the README and returned in API responses).
CPU_RUNTIME_OVERHEAD_MIB = 256.0
GPU_CONTEXT_OVERHEAD_MIB = 512.0
MEMORY_HEADROOM_FRACTION = 0.90
DATASET_COPY_OVERHEAD = 1.15
ACTIVATION_BUFFER_FACTOR = 1.15
DEFAULT_CPU_GHZ = 2.0
DEFAULT_GPU_CLOCK_MHZ = 1500.0
GPU_FP32_LANES_PER_SM = 64  # conservative generic assumption, not a device lookup
TRAINING_BACKWARD_MULTIPLIER = 2.0  # backward FLOPs ~= 2x forward FLOPs

DISCLAIMER = (
    "These are heuristic estimates for planning, not guarantees. Real training "
    "throughput depends on model implementation, framework versions, data "
    "pipeline, thermal limits, shared system load, and driver behavior. "
    "Benchmark results are measured on this host and are the most reliable "
    "numbers in this application. Hardware facts are detected, never invented; "
    "unavailable values are reported as null or with an explicit estimate label."
)

LOG_LEVEL = os.getenv("MLCP_LOG_LEVEL", "INFO").upper()
