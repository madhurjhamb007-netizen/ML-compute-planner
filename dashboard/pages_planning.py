"""Dashboard pages: model configuration, compute, memory, and time prediction."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import APIError, PlannerClient
from dashboard.components import (
    AMBER,
    COLORWAY,
    CYAN,
    GREEN,
    PRIMARY,
    VIOLET,
    badge,
    callout,
    config_chip,
    fit_badge,
    gauge_figure,
    human_count,
    human_flops,
    human_mb,
    human_seconds,
    page_header,
    render_cards,
    render_warnings,
    style_figure,
)

PRECISION_LABELS = ["fp32", "fp16", "bf16"]
DEVICE_LABELS = ["auto", "cpu", "cuda"]
OPTIMIZER_LABELS = ["adam", "adamw", "sgd"]


def render_model_configuration(
    client: PlannerClient,
    models: list[dict[str, Any]],
    estimate: dict[str, Any] | None,
    hardware: dict[str, Any],
) -> None:
    models_by_key = {model["key"]: model for model in models}
    last_config = st.session_state.get("config")
    default_key = last_config.get("model_type", "mlp") if last_config else "mlp"
    if default_key not in models_by_key:
        default_key = "mlp"

    page_header(
        "Model Configuration",
        "Describe the workload. The backend computes parameter counts, FLOPs, memory, time, and recommendations.",
        eyebrow="04 · Workload",
    )
    keys = list(models_by_key)
    selected = st.selectbox(
        "Model type",
        options=keys,
        index=keys.index(default_key),
        format_func=lambda key: models_by_key[key]["display_name"],
        key="model_type_selector",
    )
    model_info = models_by_key[selected]
    st.markdown(
        f"**{model_info['display_name']}** · {model_info['family']} &nbsp; "
        + badge("GPU supported", "green")
        if model_info["supports_gpu"]
        else f"**{model_info['display_name']}** · {model_info['family']} &nbsp; "
        + badge("CPU workload", "amber"),
        unsafe_allow_html=True,
    )
    st.caption(model_info["description"])
    if model_info.get("notes"):
        st.caption(f"Implementation note: {model_info['notes']}")

    if last_config and last_config.get("model_type") != selected:
        st.info(
            "The model selection changed. Defaults for the new model are shown; the "
            "estimate pages still display the last submitted configuration until you run a new estimate."
        )

    defaults = dict(model_info["default_config"])
    if last_config and last_config.get("model_type") == selected:
        defaults.update(last_config)

    def number_field(
        label: str,
        key: str,
        value: int,
        minimum: int,
        maximum: int,
        step: int = 1,
        help_text: str | None = None,
        column: Any = None,
    ) -> int:
        target = column or st
        return target.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=int(value),
            step=step,
            help=help_text,
            key=f"cfg_{selected}_{key}",
        )

    with st.form(f"config_form_{selected}"):
        st.markdown("#### Training setup")
        col1, col2, col3 = st.columns(3)
        dataset_size = number_field(
            "Dataset size (samples)",
            "dataset_size",
            defaults.get("dataset_size", 5000),
            1,
            10_000_000,
            500,
            "Total samples per epoch. Predictions scale linearly except for fixed overhead.",
            col1,
        )
        batch_size = number_field(
            "Batch size",
            "batch_size",
            defaults.get("batch_size", 64),
            1,
            4096,
            8,
            "Samples per optimizer step.",
            col2,
        )
        if selected == "random_forest":
            st.markdown(
                f"{badge('Tree model: epochs / precision / optimizer do not apply', 'amber')}",
                unsafe_allow_html=True,
            )
            epochs = 1
            precision = "fp32"
            optimizer = "adam"
            device = col3.selectbox(
                "Device",
                options=["cpu"],
                index=0,
                key=f"cfg_{selected}_device",
                help="Random Forest is executed on the CPU.",
            )
        else:
            epochs = number_field(
                "Epochs",
                "epochs",
                defaults.get("epochs", 10),
                1,
                10_000,
                1,
                None,
                col3,
            )
            col4, col5, col6 = st.columns(3)
            precision = col4.selectbox(
                "Training precision",
                options=PRECISION_LABELS,
                index=PRECISION_LABELS.index(defaults.get("precision", "fp32")),
                key=f"cfg_{selected}_precision",
                help="FP16 is GPU-only here; CPU falls back to FP32. BF16 on CPU is experimental.",
            )
            device = col5.selectbox(
                "Execution preference",
                options=DEVICE_LABELS,
                index=DEVICE_LABELS.index(defaults.get("device_preference", "auto")),
                key=f"cfg_{selected}_device",
                help="Auto prefers a usable CUDA GPU when the workload is estimated to fit.",
            )
            optimizer = col6.selectbox(
                "Optimizer",
                options=OPTIMIZER_LABELS,
                index=OPTIMIZER_LABELS.index(defaults.get("optimizer", "adam")),
                key=f"cfg_{selected}_optimizer",
                help="Adam/AdamW use two FP32 moment buffers; SGD uses one momentum buffer.",
            )

        st.markdown("#### Model dimensions")
        extra: dict[str, int] = {}
        if selected in ("linear_regression", "random_forest", "mlp", "lstm", "transformer"):
            cols = st.columns(3)
            extra["features"] = number_field(
                "Input features / channels",
                "features",
                defaults.get("features", 32),
                1,
                8192,
                8,
                "Feature dimension per sample (or per token for sequence models).",
                cols[0],
            )
            if selected in ("mlp", "lstm", "transformer", "cnn"):
                extra["hidden_size"] = number_field(
                    "Hidden size / width",
                    "hidden_size",
                    defaults.get("hidden_size", 128),
                    1,
                    8192,
                    16,
                    "Width of hidden layers or Transformer d_model.",
                    cols[1],
                )
                extra["num_layers"] = number_field(
                    "Layers",
                    "num_layers",
                    defaults.get("num_layers", 2),
                    1,
                    64,
                    1,
                    None,
                    cols[2],
                )
            if selected in ("linear_regression", "random_forest", "mlp", "cnn", "lstm", "transformer"):
                extra["num_classes"] = number_field(
                    "Classes / output dimension",
                    "num_classes",
                    defaults.get("num_classes", 10),
                    2,
                    10_000,
                    1,
                    "Linear regression always emits one value.",
                    st.columns(3)[0] if selected == "linear_regression" else None,
                )
            if selected == "random_forest":
                cols = st.columns(3)
                extra["n_estimators"] = number_field(
                    "Trees (n_estimators)",
                    "n_estimators",
                    defaults.get("n_estimators", 100),
                    1,
                    2000,
                    10,
                    None,
                    cols[0],
                )
                extra["max_depth"] = number_field(
                    "Max tree depth",
                    "max_depth",
                    defaults.get("max_depth", 10),
                    1,
                    64,
                    1,
                    None,
                    cols[1],
                )
            if selected == "lstm":
                extra["sequence_length"] = number_field(
                    "Sequence length",
                    "sequence_length",
                    defaults.get("sequence_length", 64),
                    1,
                    8192,
                    8,
                    None,
                    st.columns(3)[1],
                )
            if selected == "transformer":
                cols = st.columns(3)
                extra["sequence_length"] = number_field(
                    "Sequence length",
                    "sequence_length",
                    defaults.get("sequence_length", 64),
                    1,
                    8192,
                    8,
                    None,
                    cols[1],
                )
                extra["num_heads"] = number_field(
                    "Attention heads",
                    "num_heads",
                    defaults.get("num_heads", 4),
                    1,
                    64,
                    1,
                    "Hidden size must be divisible by attention heads.",
                    cols[2],
                )
            if selected == "cnn":
                cols = st.columns(3)
                extra["in_channels"] = number_field(
                    "Input channels",
                    "in_channels",
                    defaults.get("in_channels", 3),
                    1,
                    16,
                    1,
                    None,
                    cols[0],
                )
                extra["image_size"] = number_field(
                    "Image size (square)",
                    "image_size",
                    defaults.get("image_size", 32),
                    16,
                    1024,
                    16,
                    None,
                    cols[1],
                )
                extra["base_channels"] = number_field(
                    "Base convolution channels",
                    "base_channels",
                    defaults.get("base_channels", 32),
                    1,
                    1024,
                    8,
                    "Each block doubles the channel count.",
                    cols[2],
                )

        submitted = st.form_submit_button(
            "Estimate resources", type="primary", width="stretch"
        )

    if submitted:
        config = dict(defaults)
        config.update(
            {
                "model_type": selected,
                "dataset_size": dataset_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "precision": precision,
                "device_preference": device,
                "optimizer": optimizer,
            }
        )
        config.update(extra)
        if selected == "linear_regression":
            config["num_classes"] = max(2, int(config.get("num_classes", 2)))
        if selected == "transformer":
            if config["hidden_size"] % config["num_heads"] != 0:
                st.error("Hidden size must be divisible by the number of attention heads.")
                return
            if config["num_heads"] > config["hidden_size"]:
                st.error("Attention heads cannot exceed hidden size.")
                return
        try:
            with st.spinner("Estimating parameters, FLOPs, memory, and training time..."):
                result = client.estimate(config)
        except APIError as exc:
            st.error(str(exc))
            return
        st.session_state.config = config
        st.session_state.estimate = result
        st.session_state.benchmark = None
        st.session_state.estimate_stale = False
        st.success("Estimate updated. Explore the compute, memory, time, and recommendations pages.")

    if estimate:
        st.markdown("### Current estimate snapshot")
        config_chip(
            estimate["config"],
            models_by_key.get(estimate["config"].get("model_type"), {}).get("display_name"),
        )
        render_cards(
            [
                {
                    "label": "Parameters",
                    "value": human_count(estimate["compute"]["parameter_count"]),
                    "helper": "Analytical count from the implemented architecture",
                    "tone": "primary",
                },
                {
                    "label": "Training workload",
                    "value": human_flops(estimate["compute"]["total_training_flops"]),
                    "helper": "Approximate FLOPs across all epochs",
                    "tone": "cyan",
                },
                {
                    "label": "Target device",
                    "value": estimate["time"]["target_device"].upper(),
                    "helper": "Resolved from preference, CUDA availability, and memory fit",
                    "tone": "violet",
                },
                {
                    "label": "Predicted training time",
                    "value": human_seconds(estimate["time"]["predicted_seconds"]),
                    "helper": f"Range {human_seconds(estimate['time']['predicted_low_seconds'])} – "
                    f"{human_seconds(estimate['time']['predicted_high_seconds'])}",
                    "tone": "amber",
                },
            ]
        )
        render_warnings(estimate.get("warnings", []), "Estimator warnings")
    else:
        st.info("Submit the form to generate an estimate. You can also explore hardware pages now.")


def render_compute(estimate: dict[str, Any] | None, models_by_key: dict[str, Any]) -> None:
    page_header(
        "Compute Estimation",
        "Parameters and FLOPs are analytical estimates for the implemented architecture, not measured values.",
        eyebrow="05 · Compute",
    )
    if not estimate:
        st.info("Run an estimate in Model Configuration first.")
        return
    compute = estimate["compute"]
    config_chip(
        estimate["config"],
        models_by_key.get(estimate["config"].get("model_type"), {}).get("display_name"),
    )
    render_cards(
        [
            {
                "label": "Trainable parameters",
                "value": human_count(compute["parameter_count"]),
                "helper": f"{compute['parameter_count']:,} parameters",
                "tone": "primary",
            },
            {
                "label": "Forward FLOPs / sample",
                "value": human_flops(compute["forward_flops_per_sample"]),
                "helper": "Multiply-add counted as 2 FLOPs",
                "tone": "cyan",
            },
            {
                "label": "Training FLOPs / sample",
                "value": human_flops(compute["training_flops_per_sample"]),
                "helper": "≈ 3 × forward (forward + backward heuristic)",
                "tone": "violet",
            },
            {
                "label": "Total training workload",
                "value": human_flops(compute["total_training_flops"]),
                "helper": f"{human_count(compute['optimizer_steps'])} optimizer steps"
                if compute["optimizer_steps"]
                else "Tree fitting; no gradient steps",
                "tone": "amber",
            },
        ]
    )

    breakdown = compute.get("breakdown", [])
    if breakdown:
        st.markdown("### Component breakdown")
        left, right = st.columns(2)
        with left:
            fig = go.Figure(
                go.Bar(
                    x=[item["parameters"] for item in breakdown],
                    y=[item["name"] for item in breakdown],
                    orientation="h",
                    marker_color=PRIMARY,
                    hovertemplate="%{y}<br>%{x:,.0f} parameters<extra></extra>",
                )
            )
            fig.update_layout(title="Parameters by component", xaxis_title="Parameters")
            st.plotly_chart(style_figure(fig, height=330), width="stretch")
        with right:
            fig = go.Figure(
                go.Bar(
                    x=[item["forward_flops"] for item in breakdown],
                    y=[item["name"] for item in breakdown],
                    orientation="h",
                    marker_color=CYAN,
                    hovertemplate="%{y}<br>%{x:,.0f} FLOPs/sample<extra></extra>",
                )
            )
            fig.update_layout(
                title="Forward FLOPs per sample by component",
                xaxis_title="FLOPs per sample",
            )
            st.plotly_chart(style_figure(fig, height=330), width="stretch")

        rows = [
            {
                "Component": item["name"],
                "Parameters": human_count(item["parameters"]),
                "Forward FLOPs/sample": human_flops(item["forward_flops"]),
                "Activation elements/sample": human_count(item["activation_elements"]),
            }
            for item in breakdown
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        for note in compute.get("notes", []):
            st.caption(f"• {note}")
    callout(
        "FLOP formulas mirror the benchmark architectures. Activations, normalization, "
        "pooling, data loading, and optimizer math are excluded from the FLOP counts unless "
        "a model-specific note says otherwise. Random Forest workloads are approximate operations."
    )
    with st.expander("Estimation assumptions"):
        for assumption in estimate.get("assumptions", []):
            st.markdown(f"- {assumption}")


def render_memory(estimate: dict[str, Any] | None, models_by_key: dict[str, Any]) -> None:
    page_header(
        "Memory Analysis",
        "Analytical RAM/VRAM footprint with explicit overheads. Treat it as a planning budget, not an allocator guarantee.",
        eyebrow="06 · Memory",
    )
    if not estimate:
        st.info("Run an estimate in Model Configuration first.")
        return
    memory = estimate["memory"]
    config_chip(
        estimate["config"],
        models_by_key.get(estimate["config"].get("model_type"), {}).get("display_name"),
    )
    render_cards(
        [
            {
                "label": "CPU-only training RAM",
                "value": human_mb(memory["cpu_training_ram_mb"]),
                "helper": f"Fit: {'yes' if memory['fits_cpu'] else 'no'} against available RAM",
                "tone": "cyan",
            },
            {
                "label": "GPU training VRAM",
                "value": human_mb(memory["gpu_training_vram_mb"])
                if memory["gpu_training_vram_mb"] > 0
                else "N/A",
                "helper": (
                    "Weights + gradients + optimizer + activations + CUDA context"
                    if memory["gpu_training_vram_mb"] > 0
                    else "Tree model runs on CPU only"
                ),
                "tone": "violet",
            },
            {
                "label": "Host RAM during GPU training",
                "value": human_mb(memory["gpu_host_ram_mb"])
                if memory["gpu_host_ram_mb"] > 0
                else "N/A",
                "helper": "Dataset cache + loader buffers + runtime",
                "tone": "primary",
            },
            {
                "label": "GPU inference VRAM",
                "value": human_mb(memory["gpu_inference_vram_mb"])
                if memory["gpu_inference_vram_mb"] > 0
                else "N/A",
                "helper": "Weights + one batch + CUDA context",
                "tone": "amber",
            },
        ]
    )
    target = memory["target_device"]
    st.markdown(
        f"### Target path · {target.upper()} &nbsp; "
        + badge(
            "Fits" if memory["fits_target"] else "Does not fit",
            "green" if memory["fits_target"] else "red",
        ),
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.35])
    with left:
        utilization = memory["target_utilization_pct"]
        gauge_max = max(100.0, math.ceil(utilization / 10.0) * 10.0)
        st.plotly_chart(
            gauge_figure(
                utilization,
                f"Estimated {target.upper()} budget utilization",
                max_value=gauge_max,
                color=GREEN if memory["fits_target"] else "#ef4444",
                height=260,
            ),
            width="stretch",
        )
    with right:
        break_components = (
            memory["gpu_breakdown"] if target == "cuda" else memory["cpu_breakdown"]
        )
        if break_components:
            names = [item["name"] for item in break_components]
            values = [item["mb"] for item in break_components]
            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=names,
                    orientation="h",
                    marker_color=[COLORWAY[i % len(COLORWAY)] for i in range(len(names))],
                    hovertemplate="%{y}<br>%{x:,.1f} MiB<extra></extra>",
                )
            )
            fig.update_layout(title="Target memory breakdown", xaxis_title="Memory (MiB)")
            st.plotly_chart(style_figure(fig, height=300), width="stretch")

    st.markdown("### Estimated fit across paths")
    rows = [
        {
            "Path": "CPU training (RAM)",
            "Required": human_mb(memory["cpu_training_ram_mb"]),
            "Available": human_mb(estimate["hardware"]["ram"]["available_mb"]),
            "Fit": "Yes" if memory["fits_cpu"] else "No",
        },
        {
            "Path": "GPU training (VRAM)",
            "Required": human_mb(memory["gpu_training_vram_mb"])
            if memory["gpu_training_vram_mb"] > 0
            else "N/A",
            "Available": human_mb(
                (estimate["hardware"]["gpus"][0].get("free_memory_mb") or 0)
                if estimate["hardware"].get("gpus")
                else 0
            )
            if memory.get("gpu_training_vram_mb", 0) > 0
            else "N/A",
            "Fit": "N/A"
            if memory["fits_gpu"] is None
            else ("Yes" if memory["fits_gpu"] else "No"),
        },
        {
            "Path": "GPU training (host RAM)",
            "Required": human_mb(memory["gpu_host_ram_mb"])
            if memory["gpu_host_ram_mb"] > 0
            else "N/A",
            "Available": human_mb(estimate["hardware"]["ram"]["available_mb"]),
            "Fit": "N/A" if memory["gpu_host_ram_mb"] == 0 else "See target fit",
        },
        {
            "Path": "GPU inference (VRAM)",
            "Required": human_mb(memory["gpu_inference_vram_mb"])
            if memory["gpu_inference_vram_mb"] > 0
            else "N/A",
            "Available": human_mb(
                (estimate["hardware"]["gpus"][0].get("free_memory_mb") or 0)
                if estimate["hardware"].get("gpus")
                else 0
            )
            if memory.get("gpu_inference_vram_mb", 0) > 0
            else "N/A",
            "Fit": "—",
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("### All modeled components")
    cpu_col, gpu_col = st.columns(2)
    with cpu_col:
        if memory["cpu_breakdown"]:
            fig = go.Figure(
                go.Bar(
                    x=[item["mb"] for item in memory["cpu_breakdown"]],
                    y=[item["name"] for item in memory["cpu_breakdown"]],
                    orientation="h",
                    marker_color=CYAN,
                    hovertemplate="%{y}<br>%{x:,.1f} MiB<extra></extra>",
                )
            )
            fig.update_layout(title="CPU RAM components", xaxis_title="MiB")
            st.plotly_chart(style_figure(fig, height=320), width="stretch")
    with gpu_col:
        if memory["gpu_breakdown"]:
            fig = go.Figure(
                go.Bar(
                    x=[item["mb"] for item in memory["gpu_breakdown"]],
                    y=[item["name"] for item in memory["gpu_breakdown"]],
                    orientation="h",
                    marker_color=VIOLET,
                    hovertemplate="%{y}<br>%{x:,.1f} MiB<extra></extra>",
                )
            )
            fig.update_layout(title="GPU VRAM components", xaxis_title="MiB")
            st.plotly_chart(style_figure(fig, height=320), width="stretch")
        else:
            st.info("No GPU memory breakdown applies to this model.")

    with st.expander("Memory notes and assumptions"):
        for note in memory.get("notes", []):
            st.markdown(f"- {note}")
    callout(
        "CUDA context, fragmentation, and autograd storage are approximations. "
        "Always monitor `torch.cuda.max_memory_allocated()` during a short warmup, and "
        "keep 10–15% headroom for runtime fragmentation and driver overhead."
    )


def render_time(estimate: dict[str, Any] | None, models_by_key: dict[str, Any]) -> None:
    page_header(
        "Training Time Prediction",
        "A heuristic throughput model with a deliberately wide uncertainty range. Benchmarking is the calibration step.",
        eyebrow="07 · Time",
    )
    if not estimate:
        st.info("Run an estimate in Model Configuration first.")
        return
    time_data = estimate["time"]
    config_chip(
        estimate["config"],
        models_by_key.get(estimate["config"].get("model_type"), {}).get("display_name"),
    )
    render_cards(
        [
            {
                "label": "Predicted training time",
                "value": human_seconds(time_data["predicted_seconds"]),
                "helper": f"{time_data['predicted_hours']:.3f} hours; point estimate",
                "tone": "primary",
            },
            {
                "label": "Uncertainty range",
                "value": f"{human_seconds(time_data['predicted_low_seconds'])} – "
                f"{human_seconds(time_data['predicted_high_seconds'])}",
                "helper": "Planning range, not a statistical confidence interval",
                "tone": "amber",
            },
            {
                "label": "Effective throughput",
                "value": f"{time_data['effective_tflops']:.3f} TFLOP/s",
                "helper": f"{human_count(time_data['throughput_samples_per_second'])} samples/s",
                "tone": "cyan",
            },
            {
                "label": "Prediction confidence",
                "value": time_data["confidence"].upper(),
                "helper": time_data["bottleneck_primary"],
                "tone": "green" if time_data["confidence"] == "medium" else "red",
            },
        ]
    )
    low, expected, high = (
        time_data["predicted_low_seconds"],
        time_data["predicted_seconds"],
        time_data["predicted_high_seconds"],
    )
    fig = go.Figure(
        go.Bar(
            x=[low, expected, high],
            y=["Low estimate", "Point estimate", "High estimate"],
            orientation="h",
            marker_color=[CYAN, PRIMARY, VIOLET],
            text=[human_seconds(low), human_seconds(expected), human_seconds(high)],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.3f} seconds<extra></extra>",
        )
    )
    fig.update_layout(title="Predicted duration range", xaxis_title="Training seconds", showlegend=False)
    st.plotly_chart(style_figure(fig, height=300), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Execution path")
        st.markdown(
            f"""
            <div class="mlcp-card" style="--tone:{PRIMARY};">
              <div class="label">Target device / precision</div>
              <div class="value">{time_data['target_device'].upper()} · {time_data['precision_used'].upper()}</div>
              <div class="helper">
                Compute time {human_seconds(time_data['compute_seconds'])} +
                overhead {human_seconds(time_data['overhead_seconds'])}.
                The overhead allowance covers startup, optimizer-step dispatch, and framework
                work, not a full production data pipeline.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        callout(
            "This is a planning estimate, not a promise. Actual duration depends on the "
            "data pipeline, kernel fusion, thermal throttling, shared load, and implementation details."
        )
    with right:
        if time_data.get("notes"):
            st.markdown("#### What drives this prediction")
            for note in time_data["notes"]:
                st.caption(f"• {note}")
    with st.expander("All estimation assumptions"):
        for assumption in estimate.get("assumptions", []):
            st.markdown(f"- {assumption}")
    st.info(
        "Run the safe benchmark on the Benchmarking page to compare predicted and measured "
        "throughput on this exact host. Use the observed ratio to calibrate your expectations."
    )
