# PROJECT GUIDE — ML Compute Planner

This guide explains how the application works internally, how the estimates are derived, where the safety boundaries are, and how to extend or study the project. For installation and user-facing instructions, start with [README.md](README.md).

---

## 1. Design goals

1. **Never invent hardware facts.** Detection failures produce warnings and explicit fallbacks, not plausible-looking fake values.
2. **Distinguish estimates from measurements.** Analytical values are labelled as heuristics; benchmarks are labelled as measurements on this host.
3. **Be useful on a GPU-less machine.** CPU-only detection, estimation, recommendations, and benchmarks must remain fully available.
4. **Bound every real workload.** A benchmark must be small, server-enforced, memory-aware, and time-aware.
5. **Keep frontend and backend genuinely connected.** The Streamlit dashboard uses HTTP and does not import estimation internals.
6. **Explain assumptions in the product.** Every important estimate exposes its basis and limitations.

---

## 2. Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Streamlit dashboard                   │
│  app.py → navigation + session state + cached API data   │
│  pages_hardware / pages_planning / pages_benchmark       │
│  api_client.py (requests)     components.py (UI/charts)  │
└──────────────────────────────┬───────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼───────────────────────────┐
│                       FastAPI backend                    │
│  main.py (app, middleware, root routes)                  │
│  api/routes.py (/api/v1/*)                               │
│                                                          │
│  hardware/detector ─┬─ cpu.py (psutil, platform)          │
│                     └─ gpu.py (NVML → PyTorch → nvidia-smi)
│  models/registry + architectures (real PyTorch modules)  │
│  estimation/flops → memory → runtime → time → recommender
│  benchmark/runner → workloads → monitor                  │
│  services/planner.py (assembly)                          │
└──────────────────────────────────────────────────────────┘
```

### Request flow: estimate

1. Dashboard posts a `ModelConfig` JSON to `POST /api/v1/estimate`.
2. FastAPI validates the payload with Pydantic.
3. `planner.estimate_plan()` detects hardware (unless a snapshot is injected by tests).
4. `estimate_compute()` builds the component breakdown, parameters, FLOPs, and steps.
5. `estimate_memory()` computes CPU RAM, GPU VRAM, host RAM, inference VRAM, and fit.
6. `predict_training_time()` resolves the device and precision, applies the heuristic throughput model, and returns a range.
7. `build_suitability()`, `build_bottlenecks()`, and `build_recommendations()` add planning advice.
8. Warnings, assumptions, timestamped hardware, and the disclaimer are assembled into `EstimateResponse`.

### Request flow: benchmark

1. Dashboard posts the flat model config plus a nested `benchmark` object.
2. `BenchmarkRequest` validation enforces hard numeric limits.
3. `build_plan()` applies further server-side caps and memory/time safety logic.
4. The workload is prepared with a real PyTorch model or a real forest fit.
5. `ResourceSampler` samples system CPU, process CPU/RSS, GPU utilization, and VRAM in a background thread.
6. The measured loop runs; CUDA is synchronized.
7. `run_benchmark()` builds a `BenchmarkResult`, predicts the same effective workload, and produces a `BenchmarkComparison`.
8. The dashboard renders timings, telemetry, comparison, warnings, and safety notes.

### Dashboard state and caching

- `get_client()` is a cached Streamlit resource keyed by API URL.
- Health and hardware are cached for 10 seconds; models for 5 minutes; estimates and benchmarks are **not** cached.
- Session state holds `config`, `estimate`, and `benchmark`.
- Submitting a new estimate clears the old benchmark.
- **Refresh hardware** clears the health/hardware cache; **Reset session** clears plan state.
- If the API is unreachable, the dashboard stops with explicit startup instructions instead of rendering partial fake data.

---

## 3. Hardware detection

### CPU

Detected with `psutil` and platform APIs:

- model name: Windows registry, `/proc/cpuinfo`, `sysctl`, or a platform fallback;
- physical/logical core counts;
- current/max frequency when exposed, with a generic 2.0 GHz fallback clearly noted;
- per-logical-thread and overall utilization;
- recommended compute threads: one physical core fewer on systems with ≥4 physical cores;
- theoretical FP32 peak: `physical_cores × reference_GHz × 16 FLOPs/cycle/core` (AVX2 FMA assumption).

This peak is an **upper bound**, not a microbenchmark. The time predictor applies family efficiency and utilization factors.

### RAM

Total, available, used, and percent utilization come from `psutil.virtual_memory()`. Estimates compare against **90% of available RAM**, leaving headroom for the OS and fragmentation.

### GPU

Detection merges three sources without overwriting richer values:

1. `nvidia-smi` CSV (fallback; may expose driver, temperature, clocks);
2. NVML via `nvidia-ml-py` (live used/free VRAM, utilization, memory-controller utilization, power, temperature, clocks, capability);
3. PyTorch CUDA properties (authoritative CUDA usability, compute capability, SM count).

A physically visible GPU is reported even if PyTorch cannot use CUDA. In that case `cuda_available=False`, and every consumer of the snapshot treats GPU execution as unavailable. When no GPU exists, `gpus=[]`; VRAM estimates remain hypothetical planning numbers and are labelled that way.

---

## 4. Model implementations and formulas

Notation:

- `F` = input features
- `H` = hidden size
- `L` = number of hidden layers (or recurrent/encoder layers)
- `C` = number of classes/output dimension
- `B` = batch size
- `N` = dataset size
- `E` = epochs
- `S` = sequence length
- `D` = Transformer `d_model`
- `A` = attention heads
- `Cin/Cout` = convolution input/output channels
- `K` = kernel size (3 in this project)
- `P` = total parameter count

### Linear Regression

Real module: `nn.Linear(F, 1)`; objective: MSE.

- Parameters: `P = F + 1`
- Forward FLOPs/sample: `2F` (the bias add is excluded from the convention)
- Output activation elements/sample: `1`

### MLP

Real module: `L` hidden `Linear + ReLU` blocks followed by a linear output.

- First hidden layer: `F·H + H`
- Each additional hidden layer: `H·H + H`
- Output layer: `H·C + C`
- `P = F·H + H + (L−1)(H² + H) + H·C + C`
- Forward FLOPs/sample: `2[F·H + (L−1)H² + H·C]`
- Activation elements/sample: one vector per layer, `L·H + C` (the analytical breakdown sums layer widths)

### CNN

Real module: three `Conv2d(3×3, padding=1) + ReLU` blocks. The first two are followed by `MaxPool2d(2)`. The third is followed by `AdaptiveAvgPool2d(4)`. A flatten, dense hidden layer, and output layer follow. Channels are `C1 = base`, `C2 = 2·base`, `C3 = 4·base`.

- Conv parameters: `Cin·C1·9 + C1`, `C1·C2·9 + C2`, `C2·C3·9 + C3`
- Flatten size: `C3 × 16` (adaptive 4×4 output)
- Dense: `16·C3·H + H`; output: `H·C + C`
- Conv forward FLOPs: `2 × Hout × Wout × Cin × Cout × 9` at each block's input resolution
- Pooling, activations, and adaptive pooling arithmetic are excluded
- **Parameter count is independent of image size** because of the adaptive pool; convolution FLOPs are not.

### LSTM

Real module: stacked `nn.LSTM(input_size=F, hidden_size=H, num_layers=L, batch_first=True)`, then `nn.Linear(H, C)` on the last hidden state.

Per LSTM layer with input size `I`:

- Parameters: `4H(I + H) + 8H` (four gates plus two biases per gate)
- Forward FLOPs across `S` timesteps: `S · 2 · 4H(I + H)`
- Head: `H·C + C` parameters and `2HC` FLOPs
- Activation elements are approximated as `S·H·L·4 + S·H·2`

### Transformer Encoder

Real module: input projection `F→D`, learned positional table `S×D`, `L` `TransformerEncoderLayer` blocks with `nhead=A`, `dim_feedforward=4D`, dropout 0.1, batch-first, mean pooling, and a linear head.

- Input projection: `F·D + D` parameters, `2SFD` FLOPs
- Positional table: `S·D` parameters
- Per encoder layer: `12D² + 13D` parameters
- Per encoder layer FLOPs: `24SD² + 4S²D` (projections + MLP + attention score/value products)
- Head: `D·C + C` parameters, `2DC` FLOPs
- Attention memory/activations scale with `S²`; this is explicitly noted in the UI and bottlenecks.

The analytical parameter count is tested against the actual PyTorch module. Softmax, layer norm, residual adds, dropout, and bias adds are excluded from the FLOP convention.

### Random Forest

Real benchmark path: scikit-learn `RandomForestClassifier` when available; otherwise `MiniRandomForest`, a real bootstrap + recursive split-search NumPy implementation.

Estimates are approximations:

- Effective depth: `min(max_depth, ceil(log2(max(2, N))))`
- Max nodes/tree: `min(2^(depth+1) − 1, 2N − 1)`
- Parameter-like scalars: `trees × nodes × 3` (feature/threshold/leaf approximation)
- Forward traversal FLOPs/sample: `trees × depth × 2`
- Training work estimate: `trees × N × depth × sqrt(F) × 4`
- Epochs, precision, optimizer state, and GPU memory are not applicable.

The API labels these as operation approximations, not dense FLOPs.

---

## 5. Compute estimation

`ComputeEstimate` reports:

- `parameter_count` and `trainable_parameters` (identical here);
- `forward_flops_per_sample`;
- `backward_flops_per_sample = 2 × forward`;
- `training_flops_per_sample = 3 × forward`;
- `total_forward_flops = forward × N × E`;
- `total_training_flops = training × N × E`;
- `optimizer_steps = ceil(N/B) × E`;
- total activation elements and a per-component breakdown;
- notes describing exclusions and model-specific cautions.

Conventions:

- one multiply-add = 2 FLOPs;
- one forward + backward ≈ 3× forward (standard planning heuristic);
- optimizer math, data loading, loss reduction, and framework overhead are excluded from FLOPs;
- Random Forest uses operation estimates instead.

Implications:

- CNN convolution cost scales with image area and channel products.
- Transformer attention cost scales quadratically with sequence length.
- LSTM cost scales linearly with sequence length but has poor parallel efficiency on CPUs.
- MLP cost scales with width squared between hidden layers.
- Doubling dataset size or epochs doubles workload FLOPs (fixed dimensions).
- Batch size does not change FLOPs per sample, but it changes step count and utilization.

---

## 6. Memory estimation

All values are **MiB** (1 MiB = 1024² bytes).

### Neural CPU training

| Component | Basis |
|-----------|-------|
| Model weights | `P × 4 bytes` |
| Gradients | `P × 4 bytes` |
| Optimizer state | Adam/AdamW: `P × 8`; SGD momentum: `P × 4` |
| Activations (batch) | `activation_elements × B × precision_bytes × 1.15` |
| Input batch | `input_elements × B × 4 × 1.1` |
| Dataset (host) | `N × input_elements × 4 × 1.15` |
| Runtime + buffering | `256 MiB + 15% of the payload` |

CPU fit compares the total against **90% of currently available RAM**.

### Neural GPU training

| Component | Basis |
|-----------|-------|
| Weights | FP32 master weights (`P × 4`) |
| Gradients | FP32 (`P × 4`) |
| Optimizer state | Adam/AdamW 8 bytes; SGD 4 bytes per parameter |
| Activations | `activation_elements × B × GPU precision bytes × 1.15` |
| Input batch | as CPU |
| CUDA context + workspace | `512 MiB + 10% of the payload` |

Host RAM during GPU training includes the materialized dataset, doubled loader/input buffers, and the Python runtime overhead. GPU fit requires both VRAM and host RAM to fit within 90% of what is available. The 512 MiB CUDA context is a **generic allowance**, not a queried device property.

### GPU inference

`weights (precision-sized) + 0.5 × activation buffer + input batch + 512 MiB context`.

### Precision

- FP16/BF16 activation bytes = 2; FP32 = 4.
- Weights/gradients/optimizer remain modeled at FP32 master precision because AMP keeps FP32 master weights and optimizer state.
- CPU FP16 is resolved to FP32 before memory/time estimation.
- CPU BF16 is allowed but is not assumed faster.

### Fit semantics

- `fits_cpu`: CPU path fits available RAM with headroom.
- `fits_gpu`: `None` when no usable CUDA GPU exists; otherwise whether VRAM and host RAM fit.
- `fits_target`: fit on the resolved target device.
- Target utilization is `required / currently available target capacity`; it can exceed 100%, and the gauge expands its scale rather than clamping misleadingly.
- `target_utilization_pct` uses full available capacity, while the fit threshold is 90% of it. A gauge near 90–100% can therefore already be marked as tight/not fitting.

---

## 7. Training-time prediction

### CPU path

```text
planning_peak = estimated_fp32_gflops × (recommended_threads / physical_cores)^0.9
effective_flops = planning_peak × family_efficiency × batch_factor × size_factor
compute_seconds = total_training_flops / effective_flops
overhead_seconds = 0.4 s startup + optimizer_steps × 2.5 ms
```

### CUDA path

```text
planning_peak = SMs × 64 FP32 lanes × 2 FLOPs × clock_MHz × 10^6 / 10^12   [TFLOP/s]
effective_flops = planning_peak × family_efficiency × batch_factor
                  × size_factor × precision_factor
precision_factor = 1.25 for FP16/BF16, else 1.0
compute_seconds = total_training_flops / effective_flops
overhead_seconds = 1.0 s startup + optimizer_steps × 4 ms
```

`batch_factor`:

- CPU: `clamp(0.40 + 0.60 × B/64, 0.30, 1.0)`
- CUDA: `clamp(0.40 + 0.60 × B/128, 0.30, 1.0)`

`size_factor = clamp(0.55 + 0.15 × log10(P), 0.45, 1.0)`.

Family efficiency assumptions:

| Family | CPU | CUDA |
|--------|-----|------|
| Linear Regression | 0.50 | 0.18 |
| MLP | 0.32 | 0.26 |
| CNN | 0.28 | 0.35 |
| LSTM | 0.16 | 0.20 |
| Transformer | 0.22 | 0.30 |

### Random Forest path

```text
work = trees × N × depth × sqrt(F)
compute_seconds = work / (recommended_threads × 1.8×10^7)
overhead_seconds = 0.10 + trees × 0.003
```

### Ranges

| Path | Low | High |
|------|-----|------|
| CPU | 0.55× | 1.9× |
| CUDA | 0.50× | 2.0× |
| Random Forest | 0.35× | 2.5× |

The ranges are intentionally wide planning ranges, not statistical confidence intervals. Confidence is only `"low"` or `"medium"`; the app never claims high confidence in a heuristic.

**Why can CUDA be predicted slower than CPU?** For small models and large step counts, the assumed 4 ms per-step launch overhead plus 1 s startup can dominate the tiny GPU compute time. The model reports this honestly. Benchmarks calibrate the real outcome, and larger compute-heavy workloads favor the GPU.

---

## 8. Recommendations and bottlenecks

### Suitability

CPU and GPU scores start conceptually from memory/device feasibility and are reduced for:

- workloads > 1 PFLOP (>1e15) or > 100 TFLOP (>1e14);
- deep models when CUDA is available (CPU score) or unavailable (device bottleneck);
- <4 physical cores;
- predicted CPU time > 30 min or > 2 h;
- GPUs that are absent, unusable, or too small for the estimate;
- Random Forest on GPU (GPU score 20, no benefit);
- mixed-precision availability (+5 GPU score);
- predicted GPU time > 1 h (−8).

Scores are clamped to 0–100. They are planning heuristics, not benchmarks.

### Bottlenecks

Categories: `memory`, `compute`, `data_loading`, `batch_size`, `device`, `precision`, `cpu_cores`. Severity is `high`, `medium`, or `low`, and items are sorted high→low. Examples:

- target memory does not fit → high memory bottleneck;
- ≥85% target utilization → medium memory warning;
- transformer sequence ≥256 → medium/high quadratic-attention warning;
- batch <16 → kernel underutilization warning;
- deep model without CUDA → high device bottleneck;
- requested precision unavailable → low precision warning;
- large dataset plus small batch → possible input-pipeline bottleneck.

### Recommendations

- CPU threads: recommended physical-core count capped by logical threads.
- DataLoader workers: `clamp(physical_cores // 2, 1, 4)`; Random Forest allows up to 8.
- Recommended batch: min of safe batch, dataset size, and 256 on CUDA / 64 on CPU.
- Max safe batch solves the estimated memory budget for variable per-sample activation+input cost, keeping model/dataset/runtime fixed costs.
- Precision: BF16 if CUDA and capability ≥ 8.0; otherwise FP16 if supported; otherwise FP32.
- Memory advice covers gradient checkpointing, gradient accumulation, smaller dimensions, activation share, optimizer state, and batch reduction.
- Random Forest advice covers depth/features, process-level parallelism, and histogram boosting.
- Windows notes call out `__main__` guards for DataLoader workers (the built-in benchmark does not use workers).

---

## 9. Benchmark implementation and safety

### Plan construction (PyTorch models)

1. Resolve requested device; CUDA falls back to CPU if unavailable.
2. Resolve requested precision (with capability checks).
3. Cap samples at `min(requested, 2048, dataset_size)`.
4. Cap batch at `min(requested override or config batch, 256, sample_limit)`.
5. Cap steps at `min(requested steps, 100, available_full_steps)`.
6. Cap warmup at 5.
7. Build an effective config (`epochs=1`, actual samples/batch/precision/device).
8. Estimate compute/memory for that exact config.
9. If memory does not fit and device came from `auto` CUDA, fall back to CPU and reconsider; otherwise reject with `BenchmarkSafetyError`.
10. Predict duration; reduce steps until the estimate fits the time budget.
11. If a single step is still predicted above twice the budget, reject.
12. Return a plan with warnings and safety notes.

### Plan construction (Random Forest)

- Samples ≤1,200 and ≤dataset size.
- Features ≤256.
- Trees ≤40.
- Depth ≤10.
- One fit; no warmup.
- Memory check; predicted duration can reduce tree count to fit the budget.

### Measured execution

- Torch: seed 42, real model/optimizer/loss, synthetic batch reused; warmup excluded; measured loop synchronized on CUDA; time-budget check after each step; cleanup releases references and empties the CUDA cache.
- Random Forest: seed 42, real synthetic features/labels; scikit-learn when installed; NumPy fallback checks the deadline between trees; one timed fit.
- `ResourceSampler`: a daemon thread sampling every 50 ms, plus a final capture on stop.

### Comparison

The prediction is recomputed for the **effective** benchmark config with `include_startup=False`. It compares:

- measured/predicted duration ratio;
- percentage error (`(measured − predicted) / predicted`);
- predicted and measured samples/sec and TFLOP/s;
- whether measured lies inside the predicted range;
- an interpretation that calls out under/over-prediction and low RF confidence.

Benchmark telemetry uses system-wide CPU plus process CPU; the process value can exceed 100% on multi-threaded runs. Short loops may have sparse samples. GPU telemetry is best effort and absent from CPU runs.

### Why HTTP 400 vs 500

- `BenchmarkSafetyError` → 400, with the actionable safety reason.
- `BenchmarkExecutionError` → 500, when a permitted workload fails during execution.
- Pydantic validation → 422.

---

## 10. Testing strategy

- **Deterministic hardware fixtures** (`tests/conftest.py`) keep estimation tests independent of the host; real detection has separate consistency tests.
- **Numerical parity tests** build the real PyTorch modules and compare parameter counts with the analytical formulas across non-default dimensions.
- **Behavioral tests** check monotonicity and ordering (more data → more work, wider model → more FLOPs, larger batch → different memory/time, etc.) without requiring golden numbers that would be fragile.
- **Safety tests** assert cap enforcement, memory rejection, device fallback, and API validation.
- **Real benchmark tests** run tiny MLP and Random Forest workloads, proving the execution path is not stubbed.
- **API tests** use FastAPI `TestClient` for routes, request validation, and small benchmark calls.
- **Dashboard tests** start a real temporary uvicorn server, drive Streamlit with `AppTest`, traverse all nine pages, submit estimates for all six models, execute a UI benchmark, and verify the API-down guidance.

Run:

```bash
python -m pytest -q
```

The dashboard test binds an ephemeral local port. No external service or internet connection is required.

---

## 11. Extending the project

### Add a new model architecture

1. Add the key to `ModelType` in `app/schemas.py`.
2. Add defaults and metadata in `app/models/registry.py`.
3. Implement the real module and synthetic batch shape in `app/models/architectures.py`.
4. Add parameter and FLOP formulas in `app/estimation/flops.py`.
5. Add component breakdowns and model-specific notes.
6. Add family efficiency values in `app/estimation/time_predictor.py` if the default is unsuitable.
7. Add memory/breakdown behavior only if the new model needs special treatment.
8. Extend the dashboard form and labels in `dashboard/pages_planning.py`.
9. Add parameter-parity, scaling, memory, API, and dashboard tests.
10. Run the full suite and a small real benchmark.

### Add a new precision mode

1. Extend the `Precision` literal.
2. Update resolution rules and capability checks in `app/estimation/runtime.py`.
3. Update precision byte accounting in `app/estimation/memory.py`.
4. Add AMP/autocast handling in `TorchWorkload._setup_amp`.
5. Update UI precision options.
6. Add CPU/GPU fallback tests.

### Add a new benchmark workload

1. Implement a class with `prepare()`, `measure() -> WorkloadStats`, and `cleanup()` in `app/benchmark/workloads.py`.
2. Add planning and caps in `app/benchmark/runner.py`.
3. Add result/comparison fields if the workload needs them.
4. Add API schema fields and dashboard rendering.
5. Add safety, execution, and telemetry tests.

### Add a new API endpoint

1. Define request/response schemas in `app/schemas.py`.
2. Add the service function.
3. Add the route in `app/api/routes.py` with a response model and tags.
4. Add API tests, including validation and failure paths.
5. Connect the dashboard through `dashboard/api_client.py`.

### Add a new dashboard page

1. Keep the rendering function in a page module (or add one).
2. Import it in `dashboard/app.py`.
3. Add a nav entry and dispatch branch.
4. Reuse `components.py` formatting/cards to keep styling consistent.
5. Add a page navigation assertion to `tests/test_dashboard.py`.

---

## 12. Accuracy and limitations

- The time model is a planning heuristic. It deliberately avoids false precision.
- FLOPs exclude bias adds, activation functions, normalization, pooling, softmax, dropout, and optimizer update math.
- Parameter counts are exact for the implemented modules, not for arbitrary variants.
- Memory estimates do not model allocator fragmentation, caching allocators, cuDNN workspaces in detail, or activation checkpointing.
- CUDA peak uses a generic lane assumption; real hardware and libraries vary widely.
- Random Forest estimates are operation approximations, not dense FLOPs; scikit-learn's optimized implementation can differ substantially from the NumPy fallback.
- Benchmark workloads reuse one synthetic batch, so data loading and preprocessing are excluded. This is stated in every benchmark result.
- Short benchmarks have high variance and few telemetry samples.
- NVML may be unavailable or restricted.
- Simulation does not model multi-GPU, distributed training, model parallelism, or production data pipelines.
- Hardware detection can be limited in virtual machines (e.g., hidden turbo frequency).
- Units are MiB; labels may say "MB" for brevity.

The product's honest comparison is: **analytical prediction for planning → bounded measurement for calibration → informed decision**.

---

## 13. Suggested study path

1. **Hardware basics** — cores vs threads, clocks, cache, memory bandwidth, PCIe, NVML telemetry.
2. **Work math** — parameters, MACs, FLOPs, forward/backward ratios, optimizer steps.
3. **Memory accounting** — weights, gradients, optimizer state, activations, workspace, fragmentation, headroom.
4. **Performance** — arithmetic intensity, roofline model, occupancy, batch effects, launch overhead, thermal throttling.
5. **Precision** — FP32 vs TF32 vs FP16 vs BF16, master weights, loss scaling, capability constraints.
6. **Specific architectures** — dense/convolution/recurrent/attention cost structures.
7. **Classical ML** — tree split search, bootstrap sampling, depth/feature sampling trade-offs.
8. **Benchmarking** — warmup, synchronization, measurement noise, representative workloads, calibration.
9. **Software engineering** — typed API contracts, validation, graceful degradation, HTTP client/server boundaries, caching, test doubles.
10. **Testing numerics** — parity tests against framework modules, monotonic property tests, and real execution smoke tests.

Study questions:

- Why does doubling sequence length more than double Transformer FLOPs?
- Why is a GPU not always faster for a tiny MLP?
- Why do Adam and SGD have different memory footprints?
- Why is `3 × forward` only an approximation for training cost?
- What does the 90% headroom protect against?
- Why can activation memory be the dominant term for a CNN or Transformer?
- Why does batch size change step time but not FLOPs per sample?
- What is the difference between system CPU utilization and process CPU utilization?
- Why is a measured short-run benchmark not automatically representative of a long training run?
- What would change if gradient checkpointing were modeled?
- Which estimates are exact for the implemented architecture, and which are heuristic?

---

## 14. Operational notes

- Logs are emitted to the API console; request logging includes method, path, status, and elapsed milliseconds.
- `MLCP_LOG_LEVEL` can change log verbosity (default `INFO`).
- `MLCP_API_URL` changes the dashboard's backend endpoint.
- The API binds to `127.0.0.1:8000` by default in `python -m app`; use uvicorn directly for other bind addresses or ports.
- The dashboard is configured for headless operation in `.streamlit/config.toml`.
- There is no database, authentication, model artifact storage, or user-uploaded data.

---

## 15. Disclaimer

ML Compute Planner is an educational planning tool. It does not guarantee runtime, memory use, hardware behavior, or model quality. Treat its estimates as informed hypotheses and validate consequential decisions with representative benchmarks and profiling.
