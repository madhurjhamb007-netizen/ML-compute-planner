"""Dashboard pages: benchmark execution and optimization recommendations."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.api_client import APIError, PlannerClient
from dashboard.components import (
    AMBER,
    CYAN,
    GREEN,
    PRIMARY,
    RED,
    VIOLET,
    badge,
    callout,
    config_chip,
    fit_badge,
    gauge_figure,
    human_count,
    human_mb,
    human_seconds,
    page_header,
    render_cards,
    render_warnings,
    style_figure,
)


def render_benchmark(
    client: PlannerClient,
    config: dict[str, Any] | None,
    estimate: dict[str, Any] | None,
    models_by_key: dict[str, Any],
) -> None:
    page_header(
        "Benchmarking",
        "Run a small, bounded training workload, measure real resource use, and compare it with the prediction.",
        eyebrow="08 · Measurement",
    )
    if not config:
        st.info("Run an estimate in Model Configuration first.")
        return
    config_chip(
        config,
        models_by_key.get(config.get("model_type"), {}).get("display_name"),
    )
    callout(
        "Safety limits are enforced server-side: at most 2,048 synthetic samples, "
        "100 steps, 256 batch size, and a 15-second measured loop. The benchmark reuses "
        "one synthetic batch, so data loading and disk I/O are intentionally excluded."
    )

    with st.form("benchmark_form"):
        col1, col2, col3 = st.columns(3)
        device = col1.selectbox(
            "Device",
            options=["auto", "cpu", "cuda"],
            index=0,
            key="bench_device",
            help="CUDA requests gracefully fall back to CPU if CUDA is unavailable.",
        )
        steps = col2.slider(
            "Maximum measured steps", min_value=1, max_value=100, value=20, key="bench_steps"
        )
        samples = col3.number_input(
            "Maximum samples",
            min_value=1,
            max_value=2048,
            value=512,
            step=64,
            key="bench_samples",
            help="Total samples are capped by this value; actual steps may be fewer.",
        )
        col4, col5, col6 = st.columns(3)
        batch_override = col4.number_input(
            "Batch size override (0 = use config)",
            min_value=0,
            max_value=256,
            value=0,
            step=8,
            key="bench_batch",
        )
        warmup = col5.slider(
            "Warmup steps (excluded from timing)",
            min_value=0,
            max_value=5,
            value=2,
            key="bench_warmup",
        )
        max_seconds = col6.slider(
            "Safety time budget (seconds)",
            min_value=1,
            max_value=15,
            value=15,
            key="bench_max_seconds",
        )
        run = st.form_submit_button("Run safe benchmark", type="primary", width="stretch")

    if run:
        options = {
            "device": device,
            "steps": int(steps),
            "warmup_steps": int(warmup),
            "samples": int(samples),
            "max_seconds": float(max_seconds),
        }
        if int(batch_override) > 0:
            options["batch_size"] = int(batch_override)
        try:
            with st.spinner("Running bounded benchmark. This can take several seconds..."):
                result = client.run_benchmark(config, options)
        except APIError as exc:
            st.error(str(exc))
        else:
            st.session_state.benchmark = result
            st.success("Benchmark completed. Measurements below are real values from this host.")

    response = st.session_state.get("benchmark")
    if not response:
        st.info("No benchmark has been run in this session yet.")
        return

    result = response["result"]
    comparison = response["comparison"]
    effective = response["effective_config"]
    st.markdown("### Measured result")
    render_cards(
        [
            {
                "label": "Measured loop duration",
                "value": human_seconds(result["duration_seconds"]),
                "helper": f"{result['steps_completed']} measured steps + "
                f"{result['warmup_steps']} warmup",
                "tone": "primary",
            },
            {
                "label": "Samples per second",
                "value": human_count(result["samples_per_second"]),
                "helper": f"{human_count(result['samples_used'])} samples processed",
                "tone": "cyan",
            },
            {
                "label": "Average step time",
                "value": f"{result['avg_step_ms']:.2f} ms",
                "helper": f"{result['device_used'].upper()} · {result['precision_used'].upper()}",
                "tone": "violet",
            },
            {
                "label": "Measured FLOPs/s",
                "value": f"{result['measured_tflops']:.4f} TFLOP/s",
                "helper": "Derived from analytical FLOPs ÷ measured time",
                "tone": "amber",
            },
        ]
    )
    render_cards(
        [
            {
                "label": "System CPU utilization",
                "value": f"{result['cpu_utilization_avg_pct']:.1f}%",
                "helper": f"Peak {result['cpu_utilization_peak_pct']:.1f}% · system-wide",
                "tone": "primary",
            },
            {
                "label": "Process CPU",
                "value": f"{result['process_cpu_avg_pct']:.0f}%",
                "helper": "Can exceed 100% when multiple threads are active",
                "tone": "cyan",
            },
            {
                "label": "RAM growth",
                "value": human_mb(result["ram_delta_mb"]),
                "helper": f"Peak process RSS {human_mb(result['ram_peak_mb'])}",
                "tone": "amber",
            },
            {
                "label": "GPU utilization / VRAM",
                "value": (
                    f"{result['gpu_utilization_avg_pct']:.1f}%"
                    if result.get("gpu_utilization_avg_pct") is not None
                    else "N/A"
                ),
                "helper": (
                    f"Peak device VRAM {human_mb(result.get('vram_peak_mb'))}"
                    if result.get("vram_peak_mb") is not None
                    else "No GPU telemetry (CPU run or NVML unavailable)"
                ),
                "tone": "violet",
            },
        ]
    )

    st.markdown("### Prediction vs measurement")
    render_cards(
        [
            {
                "label": "Predicted loop time",
                "value": human_seconds(comparison["predicted_seconds"]),
                "helper": f"Range {human_seconds(comparison['predicted_low_seconds'])} – "
                f"{human_seconds(comparison['predicted_high_seconds'])}",
                "tone": "slate",
            },
            {
                "label": "Measured loop time",
                "value": human_seconds(comparison["measured_seconds"]),
                "helper": f"{comparison['time_ratio']:.2f}× the point prediction",
                "tone": "green",
            },
            {
                "label": "Prediction error",
                "value": f"{comparison['time_error_pct']:+.1f}%",
                "helper": "Positive means measured slower than predicted",
                "tone": "amber" if abs(comparison["time_error_pct"]) < 50 else "red",
            },
            {
                "label": "Within expected range",
                "value": "Yes" if comparison["within_expected_range"] else "No",
                "helper": "Range is heuristic, not a statistical interval",
                "tone": "green" if comparison["within_expected_range"] else "red",
            },
        ]
    )
    predicted = comparison["predicted_seconds"]
    low = comparison["predicted_low_seconds"]
    high = comparison["predicted_high_seconds"]
    measured = comparison["measured_seconds"]
    fig = go.Figure(
        go.Bar(
            x=[predicted, measured],
            y=["Predicted", "Measured"],
            orientation="h",
            marker_color=[PRIMARY, GREEN],
            text=[human_seconds(predicted), human_seconds(measured)],
            textposition="outside",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[high - predicted, 0],
                arrayminus=[predicted - low, 0],
                color="#94a3b8",
                thickness=1.2,
                width=6,
            ),
            hovertemplate="%{y}<br>%{x:.4f} seconds<extra></extra>",
        )
    )
    fig.update_layout(
        title="Measured duration vs heuristic prediction",
        xaxis_title="Seconds",
        showlegend=False,
    )
    st.plotly_chart(style_figure(fig, height=280), width="stretch")
    st.markdown(comparison["interpretation"])

    samples = result.get("samples", [])
    if samples:
        st.markdown("### Resource telemetry during measurement")
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[item["elapsed_seconds"] for item in samples],
                    y=[item["cpu_pct"] for item in samples],
                    mode="lines+markers",
                    name="System CPU %",
                    line=dict(color=PRIMARY, width=2),
                )
            )
            if any(item.get("gpu_util_pct") is not None for item in samples):
                fig.add_trace(
                    go.Scatter(
                        x=[item["elapsed_seconds"] for item in samples],
                        y=[item.get("gpu_util_pct") for item in samples],
                        mode="lines+markers",
                        name="GPU utilization %",
                        line=dict(color=GREEN, width=2),
                    )
                )
            fig.update_layout(
                title="CPU / GPU utilization over measured loop",
                xaxis_title="Elapsed seconds",
                yaxis_title="Utilization (%)",
            )
            st.plotly_chart(style_figure(fig, height=300), width="stretch")
        with right:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=[item["elapsed_seconds"] for item in samples],
                    y=[item["ram_mb"] for item in samples],
                    mode="lines+markers",
                    name="Process RSS (MiB)",
                    line=dict(color=AMBER, width=2),
                ),
                secondary_y=False,
            )
            if any(item.get("gpu_memory_used_mb") is not None for item in samples):
                fig.add_trace(
                    go.Scatter(
                        x=[item["elapsed_seconds"] for item in samples],
                        y=[item.get("gpu_memory_used_mb") for item in samples],
                        mode="lines+markers",
                        name="GPU device used (MiB)",
                        line=dict(color=VIOLET, width=2),
                    ),
                    secondary_y=True,
                )
            fig.update_layout(
                title="Memory telemetry",
                xaxis_title="Elapsed seconds",
            )
            fig.update_yaxes(title_text="Process RSS (MiB)", secondary_y=False)
            fig.update_yaxes(title_text="GPU used (MiB)", secondary_y=True)
            st.plotly_chart(style_figure(fig, height=300), width="stretch")
        if len(samples) < 3:
            st.caption(
                "The measured loop was very short, so only a few telemetry points were captured. "
                "Increase steps or samples for a smoother trace."
            )
    else:
        st.info("No resource samples were captured because the measured loop was extremely short.")

    st.markdown("### Effective benchmark configuration")
    st.dataframe(
        [
            {"Setting": "Model", "Value": effective["model_type"]},
            {"Setting": "Device used", "Value": result["device_used"]},
            {"Setting": "Precision requested / used", "Value": f"{result['precision_requested']} / {result['precision_used']}"},
            {"Setting": "Samples", "Value": human_count(effective["dataset_size"])},
            {"Setting": "Batch size", "Value": str(result["batch_size"])},
            {"Setting": "Steps completed", "Value": str(result["steps_completed"])},
            {"Setting": "Trees (Random Forest)", "Value": str(effective.get("n_estimators")) if effective["model_type"] == "random_forest" else "N/A"},
            {"Setting": "Training FLOPs performed", "Value": f"{human_count(result['training_flops_performed'])} FLOPs"},
        ],
        width="stretch",
        hide_index=True,
    )
    render_warnings(result.get("warnings", []), "Benchmark warnings")
    with st.expander("Benchmark safety notes and measurement caveats"):
        for note in response.get("safety_notes", []):
            st.markdown(f"- {note}")
        st.markdown(f"- {response.get('disclaimer', '')}")


def render_recommendations(estimate: dict[str, Any] | None, models_by_key: dict[str, Any]) -> None:
    page_header(
        "Optimization Recommendations",
        "Actionable starting points derived from the estimate, detected hardware, and memory fit.",
        eyebrow="09 · Optimization",
    )
    if not estimate:
        st.info("Run an estimate in Model Configuration first.")
        return
    recommendations = estimate["recommendations"]
    suitability = estimate["suitability"]
    config_chip(
        estimate["config"],
        models_by_key.get(estimate["config"].get("model_type"), {}).get("display_name"),
    )
    render_cards(
        [
            {
                "label": "Recommended device",
                "value": recommendations["recommended_device"].upper(),
                "helper": "Based on support, memory fit, and workload size",
                "tone": "primary",
            },
            {
                "label": "CPU threads",
                "value": f"{recommendations['cpu_threads']}",
                "helper": "Set via torch.set_num_threads()",
                "tone": "cyan",
            },
            {
                "label": "DataLoader workers",
                "value": f"{recommendations['dataloader_workers']}",
                "helper": "Keep total workers near physical cores",
                "tone": "violet",
            },
            {
                "label": "Recommended batch",
                "value": f"{recommendations['batch_size']}",
                "helper": f"Estimated max safe: {recommendations['max_safe_batch_size']}",
                "tone": "green",
            },
            {
                "label": "Precision",
                "value": recommendations["precision"].upper(),
                "helper": (
                    "Mixed precision available"
                    if recommendations["mixed_precision_available"]
                    else "Mixed precision not available on this path"
                ),
                "tone": "amber",
            },
        ]
    )

    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(
            gauge_figure(
                suitability["cpu_score"],
                "CPU suitability score",
                color=GREEN if suitability["cpu_score"] >= 60 else AMBER,
            ),
            width="stretch",
        )
        st.markdown(
            f"CPU fit: {fit_badge(suitability['cpu_fit'])}",
            unsafe_allow_html=True,
        )
    with middle:
        st.plotly_chart(
            gauge_figure(
                suitability["gpu_score"],
                "GPU suitability score",
                color=GREEN if suitability["gpu_score"] >= 60 else AMBER,
            ),
            width="stretch",
        )
        st.markdown(
            f"GPU fit: {fit_badge(suitability['gpu_fit'])}",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Summary")
        st.markdown(suitability["summary"])
        for reason in suitability.get("reasons", []):
            st.caption(f"• {reason}")

    st.markdown("### Bottleneck analysis")
    bottlenecks = estimate.get("bottlenecks", [])
    if bottlenecks:
        severity_tone = {"high": "red", "medium": "amber", "low": "slate"}
        for item in bottlenecks:
            st.markdown(
                f"{badge(item['severity'].upper(), severity_tone[item['severity']])} "
                f"**{item['category'].replace('_', ' ').title()}** — {item['message']}",
                unsafe_allow_html=True,
            )
    else:
        st.success("No dominant bottleneck was identified.")

    st.markdown("### Memory and throughput optimizations")
    optimizations = recommendations.get("memory_optimizations", [])
    if optimizations:
        for item in optimizations:
            st.markdown(f"- {item}")
    else:
        st.success("No memory-specific warnings for this configuration.")
    st.markdown("### General advice")
    for item in recommendations.get("general_advice", []):
        st.markdown(f"- {item}")
    if recommendations.get("notes"):
        st.markdown("### Notes")
        for item in recommendations["notes"]:
            st.caption(f"• {item}")
    callout(
        "Recommendations are heuristics grounded in detected hardware and the current "
        "configuration. Start with a safe benchmark, then tune batch size and precision "
        "against measured throughput."
    )
