"""Dashboard pages: hardware overview, CPU analysis, GPU analysis."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    AMBER,
    CYAN,
    GREEN,
    PRIMARY,
    RED,
    badge,
    callout,
    fit_badge,
    gauge_figure,
    human_mb,
    human_mhz,
    page_header,
    render_cards,
    render_warnings,
    style_figure,
)


def render_hardware_overview(hardware: dict[str, Any]) -> None:
    cpu = hardware.get("cpu", {})
    ram = hardware.get("ram", {})
    gpus = hardware.get("gpus", [])
    page_header(
        "Hardware Overview",
        "Live facts detected from this machine. Missing values are reported rather than guessed.",
        eyebrow="01 · Hardware",
    )
    gpu_name = gpus[0]["name"] if gpus else "CPU only"
    render_cards(
        [
            {
                "label": "CPU",
                "value": f"{cpu.get('physical_cores', 0)}C / {cpu.get('logical_threads', 0)}T",
                "helper": cpu.get("name", "Unknown CPU"),
                "tone": "primary",
            },
            {
                "label": "Available RAM",
                "value": human_mb(ram.get("available_mb")),
                "helper": f"{human_mb(ram.get('used_mb'))} used of {human_mb(ram.get('total_mb'))}",
                "tone": "cyan",
            },
            {
                "label": "Primary GPU",
                "value": gpu_name,
                "helper": (
                    f"{human_mb(gpus[0].get('total_memory_mb'))} VRAM"
                    if gpus
                    else "No NVIDIA GPU detected"
                ),
                "tone": "violet" if gpus else "slate",
            },
            {
                "label": "CUDA for PyTorch",
                "value": "Available" if hardware.get("cuda_available") else "Not available",
                "helper": (
                    f"PyTorch {hardware.get('torch_version') or '—'} · CUDA "
                    f"{hardware.get('torch_cuda_version') or '—'}"
                ),
                "tone": "green" if hardware.get("cuda_available") else "amber",
            },
        ]
    )

    st.markdown("### Live utilization")
    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(
            gauge_figure(
                cpu.get("utilization_pct", 0),
                "CPU utilization (system)",
                color=PRIMARY,
            ),
            width="stretch",
        )
    with middle:
        st.plotly_chart(
            gauge_figure(
                ram.get("utilization_pct", 0),
                "System RAM utilization",
                color=CYAN,
            ),
            width="stretch",
        )
    with right:
        if gpus:
            gpu = gpus[0]
            used = gpu.get("used_memory_mb") or 0.0
            total = gpu.get("total_memory_mb") or 1.0
            st.plotly_chart(
                gauge_figure(
                    used / total * 100 if total else 0,
                    "Primary GPU VRAM used",
                    color=GREEN,
                ),
                width="stretch",
            )
        else:
            st.markdown(
                '<div class="mlcp-card" style="--tone:#94a3b8; min-height:190px;">'
                '<div class="label">GPU telemetry</div>'
                '<div class="value" style="font-size:1.1rem;">CPU-only mode</div>'
                '<div class="helper">GPU detection, hypothetical VRAM estimates, and '
                "CPU benchmarks remain available. CUDA benchmarks require a working "
                "NVIDIA driver and CUDA-enabled PyTorch.</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### System snapshot")
    rows = [
        {"Property": "Hostname", "Value": hardware.get("hostname", "—")},
        {"Property": "Operating system", "Value": hardware.get("os", "—")},
        {"Property": "Python", "Value": hardware.get("python_version", "—")},
        {"Property": "PyTorch", "Value": hardware.get("torch_version") or "not available"},
        {"Property": "PyTorch CUDA build", "Value": hardware.get("torch_cuda_version") or "—"},
        {"Property": "NVIDIA driver", "Value": hardware.get("nvidia_driver_version") or "not detected"},
        {"Property": "CPU architecture", "Value": cpu.get("architecture", "—")},
        {"Property": "Detection time", "Value": f"{hardware.get('detection_seconds', 0):.2f} s"},
        {"Property": "Snapshot (UTC)", "Value": hardware.get("detected_at", "—")},
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    render_warnings(hardware.get("warnings", []), "Hardware detection notes")
    callout(
        "Hardware values are detected live. Utilization changes between requests. "
        "Theoretical peak throughput is computed from core count and reported frequency, "
        "not from a hardware database."
    )


def render_cpu_analysis(hardware: dict[str, Any]) -> None:
    cpu = hardware.get("cpu", {})
    page_header(
        "CPU Analysis",
        "Detected cores, threads, frequency, live utilization, and a conservative threading plan.",
        eyebrow="02 · CPU",
    )
    render_cards(
        [
            {
                "label": "Physical cores",
                "value": f"{cpu.get('physical_cores', 0)}",
                "helper": "Used for planning peak compute and thread recommendation",
                "tone": "primary",
            },
            {
                "label": "Logical threads",
                "value": f"{cpu.get('logical_threads', 0)}",
                "helper": cpu.get("architecture", "—"),
                "tone": "cyan",
            },
            {
                "label": "Recommended threads",
                "value": f"{cpu.get('recommended_threads', 0)}",
                "helper": "Leaves capacity for the OS on 4+ physical cores",
                "tone": "green",
            },
            {
                "label": "Current / max clock",
                "value": f"{human_mhz(cpu.get('current_frequency_mhz'))}",
                "helper": f"Max reported: {human_mhz(cpu.get('max_frequency_mhz'))}",
                "tone": "violet",
            },
        ]
    )

    per_core = cpu.get("per_core_utilization_pct", [])
    if per_core:
        colors = [
            RED if value >= 85 else AMBER if value >= 60 else CYAN if value >= 25 else "#cbd5e1"
            for value in per_core
        ]
        fig = go.Figure(
            go.Bar(
                x=[f"CPU {index}" for index in range(len(per_core))],
                y=per_core,
                marker_color=colors,
                hovertemplate="%{x}<br>%{y:.1f}% utilized<extra></extra>",
            )
        )
        fig.update_layout(
            title="Per-logical-thread CPU utilization",
            yaxis=dict(range=[0, 100], title="Utilization (%)"),
            xaxis_title="Logical thread",
            showlegend=False,
        )
        st.plotly_chart(style_figure(fig, height=320), width="stretch")
    else:
        st.info("Per-core utilization is unavailable on this platform.")

    left, right = st.columns([1, 1.6])
    with left:
        st.plotly_chart(
            gauge_figure(cpu.get("utilization_pct", 0), "Overall CPU utilization", color=PRIMARY),
            width="stretch",
        )
    with right:
        st.markdown("#### Throughput planning assumptions")
        st.markdown(
            f"""
            <div class="mlcp-card" style="--tone:{PRIMARY};">
              <div class="label">Estimated theoretical FP32 peak</div>
              <div class="value">{cpu.get('estimated_fp32_gflops', 0):,.1f} GFLOP/s</div>
              <div class="helper">
                Derived as physical cores × reported GHz × 16 FLOPs/cycle/core
                (AVX2 FMA assumption). This is an upper bound, not a measured
                benchmark. Time predictions apply family-specific efficiency factors.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Threading plan:** use `torch.set_num_threads({cpu.get('recommended_threads', 1)})` "
            f"on a {cpu.get('physical_cores', 1)}-physical-core CPU. Avoid starting more "
            "compute threads than physical cores; reserve capacity for the OS and data loading."
        )
    with st.expander("CPU detection details and notes"):
        for note in cpu.get("notes", []):
            st.markdown(f"- {note}")
        st.json(
            {
                "name": cpu.get("name"),
                "architecture": cpu.get("architecture"),
                "current_frequency_mhz": cpu.get("current_frequency_mhz"),
                "max_frequency_mhz": cpu.get("max_frequency_mhz"),
            }
        )


def render_gpu_analysis(hardware: dict[str, Any], estimate: dict[str, Any] | None) -> None:
    gpus = hardware.get("gpus", [])
    page_header(
        "GPU Analysis",
        "Driver-level GPU facts and live telemetry. No values are fabricated when no GPU is present.",
        eyebrow="03 · GPU",
    )
    if not gpus:
        st.warning("No NVIDIA GPU is visible to this system.")
        callout(
            "CPU-only functionality is fully available: estimates, memory planning, "
            "recommendations, and CPU benchmarks all work. Hypothetical VRAM requirements "
            "are still calculated so you can plan a future GPU."
        )
        if estimate:
            memory = estimate.get("memory", {})
            suitability = estimate.get("suitability", {})
            render_cards(
                [
                    {
                        "label": "Hypothetical training VRAM",
                        "value": human_mb(memory.get("gpu_training_vram_mb")),
                        "helper": "Weights + gradients + optimizer + activations + context",
                        "tone": "violet",
                    },
                    {
                        "label": "Hypothetical inference VRAM",
                        "value": human_mb(memory.get("gpu_inference_vram_mb")),
                        "helper": "One batch at the configured precision",
                        "tone": "cyan",
                    },
                    {
                        "label": "GPU suitability score",
                        "value": f"{suitability.get('gpu_score', 0)}/100",
                        "helper": "Zero because no usable CUDA GPU is detected",
                        "tone": "slate",
                    },
                ]
            )
        st.markdown(
            "**To enable GPU execution:** install an NVIDIA driver, confirm `nvidia-smi` works, "
            "and install a CUDA-enabled PyTorch build. The dashboard will detect it automatically "
            "on the next hardware refresh."
        )
        render_warnings(hardware.get("warnings", []), "GPU detection notes")
        return

    for gpu in gpus:
        st.markdown(f"### GPU {gpu.get('index', 0)} · {gpu.get('name', 'Unknown GPU')}")
        chips = [
            badge(gpu.get("source", "unknown source"), "primary"),
            badge(
                "CUDA usable" if gpu.get("cuda_available") else "CUDA not usable",
                "green" if gpu.get("cuda_available") else "red",
            ),
            badge(
                f"Compute capability {gpu.get('compute_capability') or 'unknown'}",
                "cyan",
            ),
        ]
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        render_cards(
            [
                {
                    "label": "Total VRAM",
                    "value": human_mb(gpu.get("total_memory_mb")),
                    "helper": f"Free {human_mb(gpu.get('free_memory_mb'))}",
                    "tone": "violet",
                },
                {
                    "label": "VRAM used",
                    "value": human_mb(gpu.get("used_memory_mb")),
                    "helper": f"{gpu.get('memory_utilization_pct', 0):.1f}% memory-controller utilization"
                    if gpu.get("memory_utilization_pct") is not None
                    else "Memory-controller utilization unavailable",
                    "tone": "amber",
                },
                {
                    "label": "Streaming multiprocessors",
                    "value": str(gpu.get("multiprocessors") or "unknown"),
                    "helper": f"SM clock {human_mhz(gpu.get('clock_mhz'))}",
                    "tone": "primary",
                },
                {
                    "label": "Temperature",
                    "value": (
                        f"{gpu.get('temperature_c'):.0f} °C"
                        if gpu.get("temperature_c") is not None
                        else "—"
                    ),
                    "helper": (
                        f"Power draw {gpu.get('power_w'):.0f} W"
                        if gpu.get("power_w") is not None
                        else "Telemetry not exposed"
                    ),
                    "tone": "cyan",
                },
            ]
        )
        left, middle = st.columns(2)
        with left:
            st.plotly_chart(
                gauge_figure(
                    gpu.get("utilization_pct") or 0,
                    "GPU compute utilization",
                    color=GREEN,
                ),
                width="stretch",
            )
        with middle:
            total = gpu.get("total_memory_mb") or 1.0
            used = gpu.get("used_memory_mb") or 0.0
            st.plotly_chart(
                gauge_figure(used / total * 100, "VRAM utilization", color=AMBER),
                width="stretch",
            )
        with st.expander("GPU telemetry details and notes"):
            st.json(
                {
                    key: gpu.get(key)
                    for key in (
                        "index",
                        "name",
                        "total_memory_mb",
                        "used_memory_mb",
                        "free_memory_mb",
                        "utilization_pct",
                        "compute_capability",
                        "multiprocessors",
                        "clock_mhz",
                        "temperature_c",
                        "power_w",
                        "driver_version",
                        "source",
                    )
                }
            )
            for note in gpu.get("notes", []):
                st.markdown(f"- {note}")
    if not hardware.get("cuda_available"):
        st.warning(
            "A GPU is physically present, but PyTorch cannot use CUDA. Install a "
            "CUDA-enabled PyTorch build to run GPU benchmarks."
        )
    render_warnings(hardware.get("warnings", []), "GPU detection notes")
    if estimate:
        memory = estimate.get("memory", {})
        st.markdown("### Estimated requirement for the current configuration")
        render_cards(
            [
                {
                    "label": "Training VRAM",
                    "value": human_mb(memory.get("gpu_training_vram_mb")),
                    "helper": "Heuristic requirement; benchmark before committing",
                    "tone": "violet",
                },
                {
                    "label": "Inference VRAM",
                    "value": human_mb(memory.get("gpu_inference_vram_mb")),
                    "helper": "Weights + one batch + context",
                    "tone": "cyan",
                },
                {
                    "label": "GPU fit",
                    "value": "Fits" if memory.get("fits_gpu") else "Check memory",
                    "helper": f"Target utilization {memory.get('target_utilization_pct', 0):.1f}%"
                    if memory.get("target_device") == "cuda"
                    else "Current target device is CPU",
                    "tone": "green" if memory.get("fits_gpu") else "amber",
                },
            ]
        )
