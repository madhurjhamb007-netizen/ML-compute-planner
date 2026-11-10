# ML Compute Planner

**ML Compute Planner** is a hardware-aware planning workbench for ML/DL training. It detects the machine it is running on, estimates parameters, FLOPs, memory, and training time for six common workloads, recommends practical optimizations, runs small real benchmarks, and compares measured throughput with the prediction.

It is built as two connected applications:

- a **FastAPI backend** that performs detection, estimation, recommendations, and benchmarks;
- a **Streamlit dashboard** that talks to the backend over HTTP and presents all nine required analysis sections.

> Estimates are heuristic planning aids, **not guarantees**. Benchmarks are real measurements of a small bounded workload on the local machine. Hardware facts are detected or reported as unavailable; they are never invented.

---

## Highlights

- **Live hardware detection** — CPU model, physical/logical cores, clocks, utilization, RAM, NVIDIA GPU telemetry through NVML/PyTorch/nvidia-smi, CUDA build details, and graceful CPU-only operation.
- **Six workloads** — Linear Regression, Random Forest, MLP, CNN, LSTM, and Transformer Encoder.
- **Analytical estimates** — parameter counts, forward/backward/training FLOPs, optimizer steps, CPU RAM, GPU VRAM, host RAM, inference VRAM, and fit checks.
- **Honest fallbacks** — FP16 on CPU falls back to FP32; CUDA requests fall back to CPU with reasons; missing telemetry is `null`, not fabricated.
- **Heuristic training-time prediction** with a deliberately wide range and low/medium confidence.
- **Safe real benchmarks** — bounded synthetic workloads, server-enforced caps, live CPU/RAM/GPU sampling, measured samples/sec, and prediction-vs-measurement comparison.
- **Actionable recommendations** — device choice, threads, DataLoader workers, safe batch size, precision, memory optimizations, and bottlenecks.
- **Integrated dashboard** — nine sections backed by the API, with interactive Plotly charts, metric cards, warnings, and assumptions.
- **Test suite** — 83 pytest tests covering schemas, hardware fallbacks, numerical parity with PyTorch, memory/time behavior, API routes, benchmark safety, and end-to-end Streamlit rendering.

---

## Quick start

### Prerequisites

- Python **3.10+** (verified here with Python 3.14.6)
- Windows, Linux, or macOS. NVIDIA GPU telemetry requires an NVIDIA driver and NVML.
- No NVIDIA GPU is required for estimates, recommendations, or CPU benchmarks.

### 1. Install

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch from PyPI may install a CPU-only build depending on platform and version. For CUDA execution, install the PyTorch build that matches your driver from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then install the remaining requirements. The app reports the actual `torch.version.cuda` and `torch.cuda.is_available()` values it sees.

### 2. Start the API

```powershell
python -m app
```

or:

```powershell
.\scripts\run_api.ps1
```

The API listens on `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

### 3. Start the dashboard

In a second terminal, from the project root:

```powershell
streamlit run dashboard/app.py
```

or:

```powershell
.\scripts\run_dashboard.ps1
```

Streamlit normally opens `http://localhost:8501`. The dashboard expects the API at `http://127.0.0.1:8000` by default. Override it with:

```powershell
$env:MLCP_API_URL = "http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

On Linux/macOS:

```bash
export MLCP_API_URL="http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

### 4. Run the tests

```powershell
python -m pytest
```

or:

```powershell
.\scripts\run_tests.ps1
```

The dashboard tests start a temporary API server on an available local port, so no manually running server is required for pytest.

---

## Dashboard sections

| # | Section | What it shows |
|---|---------|---------------|
| 01 | **Hardware Overview** | CPU/RAM/GPU summary, live utilization gauges, system snapshot, warnings, detection time. |
| 02 | **CPU Analysis** | Per-logical-thread utilization, clocks, thread recommendation, theoretical FP32 peak and its assumptions. |
| 03 | **GPU Analysis** | Per-GPU VRAM, utilization, SMs, capability, clocks, temperature, power, CUDA usability. If no GPU exists, shows CPU-only guidance and hypothetical VRAM needs. |
| 04 | **Model Configuration** | Workload form for all six models, dimensions, batch size, epochs, precision, optimizer, and device preference. Submits an estimate to the API. |
| 05 | **Compute Estimation** | Parameters, forward/training FLOPs, universal workload, component breakdown charts/table, assumptions. |
| 06 | **Memory Analysis** | CPU RAM and GPU VRAM budgets, component breakdowns, fit across paths, headroom gauge, notes. |
| 07 | **Training Time Prediction** | Point estimate, uncertainty range, effective throughput, confidence, execution path, and drivers. |
| 08 | **Benchmarking** | Bounded real benchmark form, measured time/throughput/CPU/RAM/GPU telemetry, prediction-vs-measurement chart, safety notes. |
| 09 | **Optimization Recommendations** | Device recommendation, threads/workers/batch/precision, suitability scores, bottlenecks, memory and general advice. |

The sidebar also shows API connectivity, the active plan, a hardware summary, **Refresh hardware**, and **Reset session** controls.

---

## Supported workloads

| Model | Implementation | Key dimensions | Execution |
|-------|----------------|----------------|-----------|
| **Linear Regression** | `torch.nn.Linear(features, 1)` + MSE | features, batch, epochs | CPU/CUDA |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` when installed, real NumPy CART-style fallback | samples, features, trees, depth, classes | CPU only |
| **MLP** | Stacked dense + ReLU layers and linear output | features, hidden size, hidden layers, classes | CPU/CUDA |
| **CNN** | Three conv+ReLU+pool blocks, adaptive pool, dense classifier | channels, image size, base channels, hidden size, classes | CPU/CUDA |
| **LSTM** | Stacked LSTM encoder, last hidden state, linear head | features, sequence length, hidden size, layers, classes | CPU/CUDA |
| **Transformer Encoder** | Input projection, learned positional table, `TransformerEncoderLayer` stack, mean pool, linear head | features, sequence length, d_model, heads, layers, classes | CPU/CUDA |

**Parameter-count parity:** for every neural architecture, the analytical parameter count is tested against `sum(p.numel())` from the actual PyTorch module in `tests/test_estimation.py`.

**FLOP conventions:** one multiply-add is counted as 2 FLOPs. Bias additions, elementwise activations, normalization, pooling, dropout, and optimizer update arithmetic are excluded from the core FLOP totals. Training FLOPs are approximated as **3 × forward FLOPs** (forward + backward), a standard planning heuristic.

Random Forest has no dense-FLOP equivalent. Its numbers are an explicitly labelled approximation of split-search and traversal work; tree depth is capped by dataset size and `max_depth`.

---

## Estimation methodology

### Compute

- Counts parameters and forward FLOPs per component (dense, convolution, recurrent, attention, projection, head).
- Computes backward FLOPs as `2 × forward` and training FLOPs as `3 × forward` per sample.
- Scales by `dataset_size × epochs`.
- Counts optimizer steps as `ceil(dataset_size / batch_size) × epochs`.
- Reports a component breakdown with parameters, forward FLOPs/sample, and activation elements/sample.

### Memory (MiB)

- **CPU training:** weights, gradients, optimizer state, one batch of activations, input batch, materialized dataset, runtime/buffering overhead.
- **GPU training:** weights, gradients, optimizer state, one batch of activations, input batch, generic CUDA context/workspace overhead.
- **GPU host RAM:** dataset, loader/input buffers, Python runtime overhead.
- **GPU inference:** weights, one batch of activations/input, CUDA context.
- **Fit checks** compare requirements against 90% of available RAM or free VRAM.
- Adam/AdamW optimizer state is modeled as 8 bytes/parameter; SGD momentum as 4 bytes/parameter. Master weights and gradients are FP32 in the estimates.

### Training time

The predictor combines:

1. a detected-hardware peak throughput estimate (CPU cores × reported GHz × generic lanes; GPU SMs × generic FP32 lanes × clock),
2. model-family efficiency factors,
3. batch-size and model-size utilization factors,
4. per-optimizer-step overhead and startup allowance.

It returns a wide low/high range. **A benchmark is the authoritative calibration step**, not the analytical prediction. Tiny GPU workloads can be predicted as slower than CPU because launch/setup overhead may dominate; this is reported rather than hidden.

### Device and precision resolution

- `auto` chooses CUDA only when a usable CUDA GPU is detected and the estimated workload fits in free VRAM and host RAM; otherwise CPU.
- Explicit `cuda` falls back to CPU with a warning when unavailable or not fitting.
- FP16 on CPU falls back to FP32.
- CPU BF16 is allowed but marked experimental and receives no assumed speedup.
- CUDA FP16 requires compute capability ≥ 5.3; BF16 requires ≥ 8.0. If capability is unknown, the app attempts the requested mode and says so.

---

## Benchmarking and safety

Benchmarks run **real PyTorch forward/backward/optimizer steps** or a **real tree fit**, using synthetic data and one reused batch. Data loading, disk I/O, and preprocessing are intentionally excluded and this is reported in every result.

Server-enforced limits (`app/core/config.py`):

| Limit | Value |
|-------|-------|
| Maximum synthetic samples | 2,048 |
| Maximum measured steps | 100 |
| Maximum warmup steps | 5 |
| Maximum batch size | 256 |
| Maximum measured-loop budget | 15 seconds |
| Random Forest samples | 1,200 |
| Random Forest trees | 40 |
| Random Forest depth | 10 |

Additional safeguards:

- Pydantic validates every request before execution.
- The planner checks estimated CPU/GPU memory fit and rejects unsafe runs with HTTP 400.
- Steps are reduced automatically to stay within the time budget.
- A single predicted step that is too large is rejected.
- Benchmarks reuse one synthetic batch and never write datasets to disk.
- The tree fallback checks its time budget between trees. scikit-learn fits cannot be preempted mid-call; its conservative caps and reported warnings are the protection there.

Measured values include loop duration, samples/sec, steps/sec, average step time, system and process CPU utilization, process RSS, GPU utilization/VRAM when NVML is available, and telemetry samples over time. The comparison reports:

- predicted vs measured duration,
- prediction error (positive means measured slower),
- time and throughput ratios,
- whether the measurement fell inside the heuristic range,
- an interpretation and calibration caveat.

Very short runs capture only one or a few telemetry points because the sampler interval is 50 ms; the dashboard tells you when this happens.

---

## API overview

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service metadata |
| GET | `/health` | Root health check |
| GET | `/api/v1/health` | Versioned health check |
| GET | `/api/v1/hardware` | Full hardware snapshot |
| GET | `/api/v1/models` | Six model descriptors and defaults |
| POST | `/api/v1/estimate` | Full plan: compute, memory, time, suitability, bottlenecks, recommendations |
| POST | `/api/v1/estimate/compute` | Compute-only estimate |
| POST | `/api/v1/estimate/memory` | Memory-only estimate |
| POST | `/api/v1/estimate/time` | Time-only estimate |
| POST | `/api/v1/recommendations` | Recommendations only |
| POST | `/api/v1/benchmark/run` | Safe benchmark with nested `benchmark` options |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"model_type":"mlp","dataset_size":5000,"features":32,"batch_size":64,"epochs":10,"hidden_size":128,"num_layers":2,"num_classes":10}'
```

Benchmark example:

```json
{
  "model_type": "mlp",
  "dataset_size": 512,
  "features": 32,
  "batch_size": 64,
  "epochs": 1,
  "hidden_size": 128,
  "num_layers": 2,
  "num_classes": 10,
  "benchmark": {
    "device": "auto",
    "steps": 20,
    "warmup_steps": 2,
    "samples": 512,
    "max_seconds": 15
  }
}
```

The dashboard is a genuine HTTP client of this API. It does not import backend implementation modules.

---

## Project structure

```text
ml-compute-planner/
├── app/
│   ├── __init__.py               # Package version
│   ├── __main__.py               # python -m app
│   ├── main.py                   # FastAPI app, middleware, root routes
│   ├── schemas.py                # Pydantic contracts and validation
│   ├── api/routes.py             # Versioned API endpoints
│   ├── benchmark/
│   │   ├── runner.py             # Plan, safety caps, orchestration, comparison
│   │   ├── workloads.py          # TorchWorkload, RandomForestWorkload, MiniRandomForest
│   │   └── monitor.py            # CPU/RAM/NVML sampling thread
│   ├── core/
│   │   ├── config.py             # Constants, caps, assumptions, disclaimer
│   │   └── logging.py            # Logging configuration
│   ├── estimation/
│   │   ├── flops.py              # Parameter/FLOP formulas
│   │   ├── memory.py             # RAM/VRAM budgets and fit checks
│   │   ├── runtime.py            # Device/precision fallbacks
│   │   ├── time_predictor.py     # Heuristic time ranges
│   │   └── recommender.py        # Scores, bottlenecks, recommendations
│   ├── hardware/
│   │   ├── cpu.py                # CPU, RAM, process memory
│   │   ├── gpu.py                # NVML + PyTorch + nvidia-smi merge
│   │   └── detector.py           # Consistent hardware snapshot
│   ├── models/
│   │   ├── registry.py           # Six model descriptors/defaults
│   │   └── architectures.py      # Real PyTorch model factories and batches
│   └── services/planner.py       # End-to-end plan assembly
├── dashboard/
│   ├── app.py                    # Streamlit entry point, nav, caching
│   ├── api_client.py             # Typed HTTP client
│   ├── components.py             # Theme, cards, badges, Plotly helpers
│   ├── pages_hardware.py         # Sections 01–03
│   ├── pages_planning.py         # Sections 04–07
│   └── pages_benchmark.py        # Sections 08–09
├── tests/
│   ├── conftest.py               # Deterministic CPU/GPU hardware fixtures
│   ├── test_schemas.py
│   ├── test_hardware.py
│   ├── test_estimation.py
│   ├── test_benchmark.py
│   ├── test_api.py
│   └── test_dashboard.py
├── scripts/                      # PowerShell and shell launch/test helpers
├── .streamlit/config.toml        # Dashboard theme and headless settings
├── requirements.txt
├── pytest.ini
├── PROJECT_GUIDE.md              # Architecture, formulas, extension guide, study notes
└── README.md
```

---

## Testing

```powershell
python -m pytest -q
```

The suite covers:

- Pydantic limits, extra-field rejection, Transformer head validation, and default model configs.
- Real hardware detection consistency, NVML source merging, and forced-failure fallbacks.
- Exact parameter-count parity with PyTorch for all five neural models.
- FLOP scaling, optimizer steps, training-workload arithmetic, and Random Forest behavior.
- Memory components, optimizer/precision effects, fit checks, and CPU/GPU target resolution.
- Time-range ordering, data/batch scaling, GPU throughput assumptions, and benchmark-shaped prediction.
- Recommendation bounds, suitability scores, bottleneck ordering, and full plan responses.
- API routes, validation failures, tiny real MLP and Random Forest benchmarks, and unsafe-workload rejection.
- Streamlit end-to-end: all nine pages, estimates for six models, a UI benchmark, and the API-down message (using a temporary real API server).

Some tests execute tiny real benchmarks (seconds, not minutes). Exact pass counts are environment-dependent because optional backends such as CUDA and scikit-learn are detected at runtime.

---

## Limitations and non-goals

- **Predictions are not exact.** The time model is heuristic and intentionally wide. Real throughput depends on framework versions, kernel fusion, data pipeline, thermal throttling, shared load, and implementation details.
- **No production dataset support.** Benchmarks use synthetic tensors and one reused batch. Data loading and preprocessing are not measured.
- **No distributed training.** Multi-GPU, multi-node, DDP, FSDP, and pipeline parallelism are not modeled.
- **No GPU specification database.** GPU peak throughput uses generic FP32 lane assumptions and detected SM count/clock, not a per-device lookup. It is a planning upper-bound estimate.
- **No live benchmark process control.** Once a PyTorch benchmark starts, it checks the time budget between steps; a single very slow step cannot be interrupted mid-kernel. The plan rejects configurations whose single step is too large.
- **Memory estimates are analytical.** PyTorch allocator behavior, fragmentation, cuDNN workspaces, and temporary buffers can differ from the component model.
- **Random Forest semantics differ.** Epochs, precision, and GPU settings do not apply. Its "FLOPs" are approximate operation counts, not dense algebra.
- **Telemetry is best effort.** NVML may be unavailable, blocked, or unsupported. Missing values are reported as `null`/N/A.
- **No persistence.** The application does not store hardware snapshots, estimates, or benchmarks; each session starts fresh. The dashboard caches API responses briefly for responsiveness.
- **Units.** Memory values are MiB throughout, even where the shorter "MB" label appears in UI copy.
- **Not a profiler.** It does not replace PyTorch Profiler, Nsight Systems, `torch.cuda.max_memory_allocated()`, or representative production benchmarks.

---

## Troubleshooting

**Dashboard says it cannot reach the API**
Start `python -m app` in another terminal and confirm `http://127.0.0.1:8000/api/v1/health` returns JSON. If the API uses another host/port, set `MLCP_API_URL` before launching Streamlit.

**No NVIDIA GPU is shown**
Check `nvidia-smi` in a terminal. Install/update the NVIDIA driver. Restart the API after driver changes. CPU-only functionality is intentional and fully available.

**A GPU is visible but CUDA is not usable**
The installed PyTorch build may be CPU-only or target a different CUDA runtime. Install a matching CUDA-enabled build from pytorch.org. The GPU page explains the detected state.

**`import torch` fails on Windows**
Some locked-down Windows environments block PyTorch DLLs at the OS/security-policy level. Do not weaken security controls to work around it. A CPU-only Python environment with a permitted torch build is required for real MLP/CNN/LSTM/Transformer benchmarks; estimation and hardware pages still work without torch.

**A benchmark is rejected**
The server returns HTTP 400 with the reason, usually estimated memory or a single step exceeding the time budget. Reduce batch size, image size, sequence length, model width, samples, or steps. Random Forest has separate caps for samples/trees/depth.

**Only one telemetry point appears**
The measured loop was shorter than the 50 ms sampler interval. Increase steps or samples, or accept that very short measurements have fewer samples.

**Predictions look far off**
Run the benchmark and use the measured ratio to calibrate. Tiny workloads are dominated by fixed overhead; large workloads are more representative. The uncertainty range is a planning aid, not a statistical confidence interval.

**Streamlit shows stale hardware**
Use **Refresh hardware** in the sidebar. Hardware responses are cached for 10 seconds and models for 5 minutes.

---

## Key study concepts

Hardware detection and fallbacks · CPU/GPU peak throughput · FLOPs and MACs · analytical vs measured performance · arithmetic intensity and rooflines · parameter counting · activation memory · optimizer state · mixed precision (FP16/BF16/FP32) · CUDA capability constraints · batching and gradient accumulation · GPU occupancy and launch overhead · tree-ensemble cost · sampling and benchmarking methodology · uncertainty ranges and calibration · client/server architecture · Pydantic validation · FastAPI · Streamlit session state and caching · test doubles and numerical parity tests.

See **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** for the formulas, architecture internals, extension checklist, and a structured study path.

---

## Disclaimer

This project is an educational and planning tool. It does not guarantee training time, memory use, hardware performance, or model outcomes. Verify important decisions with representative benchmarks, profiling, and real production workloads.
