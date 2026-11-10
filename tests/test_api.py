"""FastAPI endpoint tests, including small real benchmarks."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_root_and_health_endpoints(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["docs_url"] == "/docs"
    assert "dashboard" in root.json()["dashboard_hint"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    api_health = client.get("/api/v1/health")
    assert api_health.status_code == 200
    body = api_health.json()
    assert body["app"] == "ML Compute Planner"
    assert body["version"] == "0.1.0"


def test_models_endpoint_lists_six_models(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert [model["key"] for model in models] == [
        "linear_regression",
        "random_forest",
        "mlp",
        "cnn",
        "lstm",
        "transformer",
    ]
    assert models[0]["default_config"]["model_type"] == "linear_regression"


def test_hardware_endpoint_shape(client: TestClient) -> None:
    response = client.get("/api/v1/hardware")
    assert response.status_code == 200
    hardware = response.json()
    assert hardware["cpu"]["physical_cores"] >= 1
    assert hardware["ram"]["total_mb"] > 0
    assert isinstance(hardware["gpus"], list)
    assert hardware["cuda_available"] is bool(hardware["cuda_available"])
    assert hardware["detected_at"]


def test_estimate_endpoint_returns_every_section(client: TestClient) -> None:
    payload = {
        "model_type": "mlp",
        "dataset_size": 128,
        "features": 8,
        "batch_size": 16,
        "epochs": 2,
        "hidden_size": 16,
        "num_layers": 1,
        "num_classes": 3,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    for key in (
        "config",
        "hardware",
        "compute",
        "memory",
        "time",
        "recommendations",
        "suitability",
        "bottlenecks",
        "assumptions",
        "warnings",
        "disclaimer",
    ):
        assert key in body
    assert body["compute"]["parameter_count"] == 8 * 16 + 16 + 16 * 3 + 3
    assert body["time"]["predicted_seconds"] > 0
    assert body["memory"]["cpu_training_ram_mb"] > 0


@pytest.mark.parametrize(
    ("path", "expected_key"),
    [
        ("/api/v1/estimate/compute", "parameter_count"),
        ("/api/v1/estimate/memory", "cpu_training_ram_mb"),
        ("/api/v1/estimate/time", "predicted_seconds"),
        ("/api/v1/recommendations", "recommended_device"),
    ],
)
def test_focused_estimation_endpoints(client: TestClient, path: str, expected_key: str) -> None:
    response = client.post(
        path,
        json={
            "model_type": "mlp",
            "dataset_size": 64,
            "batch_size": 16,
            "features": 8,
            "hidden_size": 16,
            "num_layers": 1,
        },
    )
    assert response.status_code == 200
    assert expected_key in response.json()


def test_invalid_configuration_returns_validation_detail(client: TestClient) -> None:
    response = client.post(
        "/api/v1/estimate",
        json={"model_type": "transformer", "hidden_size": 10, "num_heads": 4},
    )
    assert response.status_code == 422
    assert "divisible" in str(response.json()["detail"])

    response = client.post(
        "/api/v1/estimate",
        json={"model_type": "mlp", "unknown_option": True},
    )
    assert response.status_code == 422

    response = client.post("/api/v1/estimate", json={"model_type": "not_a_model"})
    assert response.status_code == 422


def test_benchmark_option_bounds_are_enforced_by_the_api(client: TestClient) -> None:
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "model_type": "mlp",
            "benchmark": {"steps": 101, "samples": 16, "warmup_steps": 0},
        },
    )
    assert response.status_code == 422


def test_tiny_mlp_benchmark_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "model_type": "mlp",
            "dataset_size": 8,
            "features": 4,
            "batch_size": 4,
            "epochs": 1,
            "hidden_size": 8,
            "num_layers": 1,
            "num_classes": 2,
            "benchmark": {
                "device": "cpu",
                "steps": 1,
                "warmup_steps": 0,
                "samples": 8,
                "max_seconds": 2,
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    result = body["result"]
    assert result["model_type"] == "mlp"
    assert result["device_used"] == "cpu"
    assert result["steps_completed"] >= 1
    assert result["samples_used"] == result["steps_completed"] * result["batch_size"]
    assert result["duration_seconds"] > 0
    assert result["samples_per_second"] > 0
    assert body["comparison"]["measured_samples_per_second"] > 0
    assert body["safety_notes"]
    assert body["disclaimer"]


def test_tiny_random_forest_benchmark_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "model_type": "random_forest",
            "dataset_size": 32,
            "features": 4,
            "num_classes": 2,
            "n_estimators": 2,
            "max_depth": 2,
            "batch_size": 8,
            "benchmark": {
                "device": "cpu",
                "steps": 1,
                "warmup_steps": 0,
                "samples": 32,
                "max_seconds": 5,
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["model_type"] == "random_forest"
    assert body["result"]["device_used"] == "cpu"
    assert body["result"]["samples_used"] == 32
    assert body["result"]["duration_seconds"] > 0
    assert any("backend" in warning.lower() for warning in body["result"]["warnings"])


def test_unsafe_workload_is_rejected_with_400(client: TestClient) -> None:
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "model_type": "cnn",
            "dataset_size": 2048,
            "batch_size": 256,
            "image_size": 512,
            "in_channels": 3,
            "base_channels": 512,
            "hidden_size": 4096,
            "num_classes": 10,
            "benchmark": {
                "device": "cpu",
                "steps": 1,
                "warmup_steps": 0,
                "samples": 2048,
                "max_seconds": 1,
            },
        },
    )
    assert response.status_code == 400
    assert "safety" in response.text.lower() or "fit" in response.text.lower()


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/does-not-exist").status_code == 404
