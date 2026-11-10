"""End-to-end Streamlit dashboard tests against a real local API server."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def api_server() -> str:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("The test API server exited before becoming ready.")
            try:
                response = requests.get(f"{url}/api/v1/health", timeout=1)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("The test API server did not start in time.")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [exception.value for exception in at.exception]
    assert not at.error, [error.value for error in at.error]


def test_dashboard_pages_forms_and_benchmark(api_server: str, monkeypatch) -> None:
    monkeypatch.setenv("MLCP_API_URL", api_server)
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)

    at.run()
    _assert_clean(at)
    assert at.radio
    labels = list(at.radio[0].options)
    assert labels == [
        "01 · Hardware Overview",
        "02 · CPU Analysis",
        "03 · GPU Analysis",
        "04 · Model Configuration",
        "05 · Compute Estimation",
        "06 · Memory Analysis",
        "07 · Training Time Prediction",
        "08 · Benchmarking",
        "09 · Optimization Recommendations",
    ]

    # Every section renders without an API or UI exception, even before an estimate.
    for label in labels:
        at.radio[0].set_value(label).run()
        _assert_clean(at)

    # Submit an estimate for every supported model through the real form.
    at.radio[0].set_value("04 · Model Configuration").run()
    for model_type in (
        "linear_regression",
        "random_forest",
        "mlp",
        "cnn",
        "lstm",
        "transformer",
    ):
        selector = next(
            widget for widget in at.selectbox if widget.key == "model_type_selector"
        )
        selector.set_value(model_type).run()
        _assert_clean(at)
        submit = next(button for button in at.button if button.label == "Estimate resources")
        submit.click().run()
        _assert_clean(at)
        assert at.session_state["estimate"]["config"]["model_type"] == model_type
        assert at.session_state["estimate"]["compute"]["parameter_count"] > 0
        assert any("Estimate updated" in message.value for message in at.success)

    # Keep the fastest model for the UI benchmark.
    selector = next(
        widget for widget in at.selectbox if widget.key == "model_type_selector"
    )
    selector.set_value("mlp").run()
    submit = next(button for button in at.button if button.label == "Estimate resources")
    submit.click().run()
    _assert_clean(at)

    at.radio[0].set_value("08 · Benchmarking").run()
    _assert_clean(at)
    for slider in at.slider:
        if slider.key == "bench_steps":
            slider.set_value(1)
        elif slider.key == "bench_warmup":
            slider.set_value(0)
        elif slider.key == "bench_max_seconds":
            slider.set_value(3)
    for number in at.number_input:
        if number.key == "bench_samples":
            number.set_value(8)
        elif number.key == "bench_batch":
            number.set_value(4)
    run_button = next(button for button in at.button if button.label == "Run safe benchmark")
    run_button.click().run()
    _assert_clean(at)
    assert any("Benchmark completed" in message.value for message in at.success)
    benchmark = at.session_state["benchmark"]
    assert benchmark["result"]["steps_completed"] >= 1
    assert benchmark["result"]["duration_seconds"] > 0
    assert benchmark["result"]["samples_per_second"] > 0
    assert benchmark["comparison"]["predicted_seconds"] > 0

    # The recommendations and GPU sections still render after measurement.
    at.radio[0].set_value("09 · Optimization Recommendations").run()
    _assert_clean(at)
    at.radio[0].set_value("03 · GPU Analysis").run()
    _assert_clean(at)


def test_dashboard_shows_actionable_message_when_api_is_down(monkeypatch) -> None:
    monkeypatch.setenv("MLCP_API_URL", f"http://127.0.0.1:{_free_port()}")
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    assert not at.exception
    assert at.error
    message = " ".join(error.value for error in at.error).lower()
    assert "cannot reach the backend api" in message
    guidance = " ".join(item.value for item in at.markdown).lower()
    assert "configured api url" in guidance
    assert "python -m app" in guidance
