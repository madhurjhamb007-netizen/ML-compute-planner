"""ML Compute Planner — Streamlit dashboard entry point.

Run from the project root:

    streamlit run dashboard/app.py

The dashboard talks to the FastAPI backend over HTTP. Start the API first
with ``python -m app`` (default: http://127.0.0.1:8000), or point the
dashboard elsewhere with the ``MLCP_API_URL`` environment variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the entry script's directory on sys.path. Add the project
# root so the shared ``dashboard`` package resolves regardless of how the
# dashboard is launched.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from dashboard.api_client import APIError, PlannerClient  # noqa: E402
from dashboard.components import (  # noqa: E402
    badge,
    footer_note,
    human_seconds,
    inject_theme,
    render_warnings,
)
from dashboard.pages_benchmark import render_benchmark, render_recommendations  # noqa: E402
from dashboard.pages_hardware import (  # noqa: E402
    render_cpu_analysis,
    render_gpu_analysis,
    render_hardware_overview,
)
from dashboard.pages_planning import (  # noqa: E402
    render_compute,
    render_memory,
    render_model_configuration,
    render_time,
)

API_URL = os.getenv("MLCP_API_URL", "http://127.0.0.1:8000").rstrip("/")

NAV_ITEMS: list[tuple[str, str]] = [
    ("01 · Hardware Overview", "System snapshot"),
    ("02 · CPU Analysis", "Cores, clocks, utilization"),
    ("03 · GPU Analysis", "VRAM and CUDA readiness"),
    ("04 · Model Configuration", "Describe the workload"),
    ("05 · Compute Estimation", "Parameters and FLOPs"),
    ("06 · Memory Analysis", "RAM and VRAM budget"),
    ("07 · Training Time Prediction", "Duration and confidence"),
    ("08 · Benchmarking", "Measure this host"),
    ("09 · Optimization Recommendations", "Tune the plan"),
]

DISCLAIMER = (
    "ML Compute Planner provides heuristic planning estimates, not guarantees. "
    "Analytical models simplify real systems; always validate important decisions "
    "with a representative benchmark and monitor actual resource use."
)


# ---------------------------------------------------------------------------
# Cached API access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_client(base_url: str) -> PlannerClient:
    return PlannerClient(base_url)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health(base_url: str) -> dict[str, Any]:
    return get_client(base_url).check_health()


@st.cache_data(ttl=10, show_spinner="Detecting hardware...")
def fetch_hardware(base_url: str) -> dict[str, Any]:
    return get_client(base_url).get_hardware()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(base_url: str) -> list[dict[str, Any]]:
    return get_client(base_url).get_models()


def _clear_hardware_cache() -> None:
    fetch_hardware.clear()
    fetch_health.clear()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("estimate", None)
    st.session_state.setdefault("benchmark", None)
    st.session_state.setdefault("hardware_refreshed_at", None)


def _nav_label() -> str:
    return st.session_state.get("nav", NAV_ITEMS[0][0])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(health: dict[str, Any], hardware: dict[str, Any] | None) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="t">⚡ ML Compute Planner</div>
              <div class="s">Hardware-aware training estimates,
              memory budgets, and safe benchmarks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**API** &nbsp; {badge('connected', 'green')}  \n"
            f"<span class='small-muted'>{API_URL} · v{health.get('version', '?')}</span>",
            unsafe_allow_html=True,
        )

        st.radio(
            "Navigation",
            options=[label for label, _ in NAV_ITEMS],
            index=0,
            format_func=lambda label: label,
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()
        estimate = st.session_state.get("estimate")
        config = st.session_state.get("config")
        if config and estimate:
            model_name = config.get("model_type", "model").replace("_", " ").title()
            st.markdown("**Active plan**")
            st.markdown(
                f"<span class='small-muted'>{model_name} · "
                f"{config.get('dataset_size', 0):,} samples · "
                f"batch {config.get('batch_size', '—')}<br>"
                f"Predicted {human_seconds(estimate['time']['predicted_seconds'])} "
                f"on {estimate['time']['target_device'].upper()}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No estimate yet. Start with Model Configuration.")

        if hardware:
            cpu = hardware.get("cpu", {})
            ram = hardware.get("ram", {})
            gpus = hardware.get("gpus", [])
            st.markdown("**This host**")
            st.caption(
                f"{cpu.get('physical_cores', '?')}C / {cpu.get('logical_threads', '?')}T CPU · "
                f"{ram.get('available_mb', 0) / 1024:.1f} GiB RAM free · "
                f"{gpus[0].get('name', 'no GPU') if gpus else 'no NVIDIA GPU'}"
            )

        st.divider()
        if st.button("Refresh hardware", width="stretch"):
            _clear_hardware_cache()
            st.session_state.hardware_refreshed_at = datetime.now().isoformat(timespec="seconds")
            st.rerun()
        if st.button("Reset session", width="stretch"):
            st.session_state.config = None
            st.session_state.estimate = None
            st.session_state.benchmark = None
            st.session_state.estimate_stale = False
            st.rerun()

        if st.session_state.get("hardware_refreshed_at"):
            st.caption(f"Last manual refresh: {st.session_state.hardware_refreshed_at}")

        st.caption(
            "Estimates are heuristics. Benchmarks are measured, bounded workloads "
            "on this exact host."
        )

    return _nav_label()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ML Compute Planner",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    _init_state()

    client = get_client(API_URL)

    try:
        health = fetch_health(API_URL)
    except APIError as exc:
        st.error("The dashboard cannot reach the backend API.")
        st.markdown(
            f"**Configured API URL:** `{API_URL}`\n\n"
            f"**Error:** {exc}\n\n"
            "Start the backend from the project root, then reload this page:\n\n"
            "```powershell\npython -m app\n```\n\n"
            "Or set a different endpoint before launching Streamlit:\n\n"
            "```powershell\n$env:MLCP_API_URL = \"http://127.0.0.1:8000\"\n"
            "streamlit run dashboard/app.py\n```"
        )
        st.stop()

    try:
        with st.spinner("Loading hardware details..."):
            hardware = fetch_hardware(API_URL)
    except APIError as exc:
        st.error(f"Hardware detection failed: {exc}")
        hardware = None

    try:
        models = fetch_models(API_URL)
    except APIError as exc:
        st.error(f"Could not load the model registry: {exc}")
        st.stop()

    models_by_key = {model["key"]: model for model in models}
    nav = render_sidebar(health, hardware)

    if hardware is None:
        st.warning(
            "Hardware data is unavailable, so hardware pages and estimates cannot be "
            "rendered. Use **Refresh hardware** after resolving the API error."
        )
        footer_note(DISCLAIMER)
        st.stop()

    estimate = st.session_state.get("estimate")
    config = st.session_state.get("config")

    if nav == "01 · Hardware Overview":
        render_hardware_overview(hardware)
    elif nav == "02 · CPU Analysis":
        render_cpu_analysis(hardware)
    elif nav == "03 · GPU Analysis":
        render_gpu_analysis(hardware, estimate)
    elif nav == "04 · Model Configuration":
        render_model_configuration(client, models, estimate, hardware)
    elif nav == "05 · Compute Estimation":
        render_compute(estimate, models_by_key)
    elif nav == "06 · Memory Analysis":
        render_memory(estimate, models_by_key)
    elif nav == "07 · Training Time Prediction":
        render_time(estimate, models_by_key)
    elif nav == "08 · Benchmarking":
        render_benchmark(client, config, estimate, models_by_key)
    elif nav == "09 · Optimization Recommendations":
        render_recommendations(estimate, models_by_key)
    else:
        st.error(f"Unknown page: {nav}")

    if hardware.get("warnings") and nav not in {
        "01 · Hardware Overview",
        "02 · CPU Analysis",
        "03 · GPU Analysis",
    }:
        render_warnings(hardware.get("warnings", []), "Hardware detection notes")

    footer_note(DISCLAIMER)


if __name__ == "__main__":
    main()
