"""Reusable dashboard UI elements, formatting, and Plotly styling."""

from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st
import plotly.graph_objects as go

PRIMARY = "#6366f1"
CYAN = "#06b6d4"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
SLATE = "#64748b"
VIOLET = "#8b5cf6"
COLORWAY = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9", "#84cc16"]
TONE_COLORS = {
    "primary": PRIMARY,
    "cyan": CYAN,
    "green": GREEN,
    "amber": AMBER,
    "red": RED,
    "violet": VIOLET,
    "slate": SLATE,
}

_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg, #eef2ff 0%, #ecfeff 65%, #f0fdfa 100%);
       border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;}
.hero .eyebrow {font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em;
                text-transform: uppercase; color: #6366f1; margin-bottom: 4px;}
.hero h1 {font-size: 2rem; line-height: 1.15; margin: 0 0 6px 0; color: #0f172a;}
.hero p {margin: 0; color: #475569; font-size: 0.98rem; max-width: 950px;}
.mlcp-card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
            padding: 15px 16px 14px 16px; min-height: 112px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
            position: relative; overflow: hidden;}
.mlcp-card::before {content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--tone);}
.mlcp-card .label {font-size: 0.76rem; text-transform: uppercase; letter-spacing: .08em;
                   color: #64748b; font-weight: 700; margin-bottom: 7px;}
.mlcp-card .value {font-size: 1.42rem; font-weight: 750; color: #0f172a; line-height: 1.15;
                   overflow-wrap: anywhere;}
.mlcp-card .helper {font-size: 0.78rem; color: #64748b; margin-top: 7px; line-height: 1.35;}
.mlcp-badge {display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 0.74rem;
             font-weight: 700; letter-spacing: .02em; border: 1px solid transparent;}
.badge-green {background:#ecfdf5; color:#047857; border-color:#a7f3d0;}
.badge-amber {background:#fffbeb; color:#b45309; border-color:#fde68a;}
.badge-red {background:#fef2f2; color:#b91c1c; border-color:#fecaca;}
.badge-slate {background:#f1f5f9; color:#475569; border-color:#e2e8f0;}
.badge-primary {background:#eef2ff; color:#4338ca; border-color:#c7d2fe;}
.badge-cyan {background:#ecfeff; color:#0e7490; border-color:#a5f3fc;}
.chip-row {display:flex; flex-wrap:wrap; gap:7px; margin: 2px 0 16px 0;}
.chip {background:#ffffff; border:1px solid #e2e8f0; border-radius:999px; padding:4px 10px;
       font-size:0.76rem; color:#475569;}
.chip b {color:#0f172a;}
.callout {background:#f8fafc; border-left:4px solid #6366f1; padding:12px 15px;
          border-radius:0 10px 10px 0; color:#334155; font-size:.9rem; margin: 8px 0 14px 0;}
.small-muted {font-size:.8rem; color:#64748b;}
h2 {padding-top: .2rem;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
section[data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
.sidebar-brand {background: linear-gradient(135deg,#4f46e5,#0891b2); color: white;
                border-radius: 14px; padding: 16px 16px 14px 16px; margin-bottom: 12px;}
.sidebar-brand .t {font-size: 1.05rem; font-weight: 800; letter-spacing: .01em;}
.sidebar-brand .s {font-size: .76rem; opacity: .9; margin-top: 3px; line-height: 1.4;}
.footer-note {color:#94a3b8; font-size:.76rem; border-top:1px solid #e2e8f0;
              padding-top: 12px; margin-top: 30px; line-height:1.5;}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "ML Compute Planner") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    helper: str = "",
    tone: str = "primary",
) -> None:
    color = TONE_COLORS.get(tone, PRIMARY)
    st.markdown(
        f"""
        <div class="mlcp-card" style="--tone:{color};">
          <div class="label">{html.escape(str(label))}</div>
          <div class="value">{html.escape(str(value))}</div>
          <div class="helper">{html.escape(str(helper))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cards(cards: Iterable[dict[str, Any]], columns: int | None = None) -> None:
    items = list(cards)
    if not items:
        return
    count = columns or len(items)
    cols = st.columns(count)
    for index, item in enumerate(items):
        with cols[index % count]:
            metric_card(
                item.get("label", ""),
                item.get("value", ""),
                item.get("helper", ""),
                item.get("tone", "primary"),
            )


def badge(text: str, tone: str = "slate") -> str:
    return f'<span class="mlcp-badge badge-{tone}">{html.escape(text)}</span>'


def fit_badge(value: bool | None) -> str:
    if value is True:
        return badge("Fits", "green")
    if value is False:
        return badge("Does not fit", "red")
    return badge("N/A", "slate")


def config_chip(config: dict[str, Any], model_name: str | None = None) -> None:
    name = model_name or config.get("model_type", "model")
    items = [
        f"<b>{html.escape(str(name))}</b>",
        f"Dataset <b>{human_count(config.get('dataset_size', 0))}</b>",
        f"Batch <b>{config.get('batch_size', '—')}</b>",
        f"Epochs <b>{config.get('epochs', '—')}</b>",
        f"Precision <b>{str(config.get('precision', '—')).upper()}</b>",
        f"Device <b>{str(config.get('device_preference', 'auto')).upper()}</b>",
    ]
    if config.get("model_type") == "random_forest":
        items = [
            f"<b>{html.escape(str(name))}</b>",
            f"Samples <b>{human_count(config.get('dataset_size', 0))}</b>",
            f"Trees <b>{config.get('n_estimators', '—')}</b>",
            f"Depth <b>{config.get('max_depth', '—')}</b>",
            f"Features <b>{config.get('features', '—')}</b>",
        ]
    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{item}</span>' for item in items) + "</div>",
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def render_warnings(warnings: list[str], title: str = "Warnings") -> None:
    unique = [item for item in dict.fromkeys(warnings) if item]
    if not unique:
        return
    with st.expander(f"⚠️ {title} ({len(unique)})", expanded=len(unique) <= 2):
        for item in unique:
            st.markdown(f"- {item}")


def footer_note(disclaimer: str) -> None:
    st.markdown(
        f'<div class="footer-note">{html.escape(disclaimer)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def human_count(value: float | int) -> str:
    value = float(value or 0)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:,.0f}"


def human_flops(value: float | int) -> str:
    value = float(value or 0)
    if value >= 1e18:
        return f"{value / 1e18:.3f} E"
    if value >= 1e15:
        return f"{value / 1e15:.3f} P"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if value >= 1e3:
        return f"{value / 1e3:.2f} K"
    return f"{value:,.0f}"


def human_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    value = max(0.0, float(seconds))
    if value < 0.001:
        return f"{value * 1e6:.0f} µs"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    if value < 60:
        return f"{value:.2f} s"
    if value < 3600:
        return f"{value / 60:.1f} min"
    if value < 86400:
        return f"{value / 3600:.2f} h"
    return f"{value / 86400:.2f} days"


def human_mb(value: float | int | None) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:,.0f} MiB"


def human_mhz(value: float | int | None) -> str:
    if not value:
        return "—"
    return f"{float(value) / 1000:.2f} GHz"


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def style_figure(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Segoe UI, sans-serif", color="#334155", size=12),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def gauge_figure(
    value: float | None,
    title: str,
    max_value: float = 100.0,
    suffix: str = "%",
    color: str = PRIMARY,
    height: int = 240,
) -> go.Figure:
    if value is None:
        value = 0.0
    value = max(0.0, min(float(value), max_value))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 30, "color": "#0f172a"}},
            title={"text": title, "font": {"size": 13, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                "bar": {"color": color, "thickness": 0.65},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_value * 0.6], "color": "#f1f5f9"},
                    {"range": [max_value * 0.6, max_value * 0.85], "color": "#fef3c7"},
                    {"range": [max_value * 0.85, max_value], "color": "#fee2e2"},
                ],
            },
        )
    )
    return style_figure(fig, height=height)


def horizontal_stacked_bar(
    categories: list[str],
    component_names: list[str],
    values: dict[str, list[float]],
    title: str,
    height: int = 330,
) -> go.Figure:
    fig = go.Figure()
    for index, name in enumerate(component_names):
        fig.add_trace(
            go.Bar(
                y=categories,
                x=values.get(name, [0] * len(categories)),
                name=name,
                orientation="h",
                marker_color=COLORWAY[index % len(COLORWAY)],
                hovertemplate=f"<b>{name}</b><br>%{{x:,.1f}} MiB<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", title=title, xaxis_title="Memory (MiB)")
    return style_figure(fig, height=height)


def empty_state(message: str, icon: str = "ℹ️") -> None:
    st.info(f"{icon} {message}")
