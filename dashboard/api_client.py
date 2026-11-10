"""Typed HTTP client for the ML Compute Planner API."""

from __future__ import annotations

import os
from typing import Any

import requests


class APIError(RuntimeError):
    """A user-facing API or connection error."""


class PlannerClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("MLCP_API_URL", "http://127.0.0.1:8000")
        ).rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
        except requests.ConnectionError as exc:
            raise APIError(
                f"Cannot reach the API at {self.base_url}. Start it with "
                "`python -m app` or `uvicorn app.main:app --port 8000`."
            ) from exc
        except requests.Timeout as exc:
            raise APIError("The API request timed out. The benchmark may still be running.") from exc
        except requests.RequestException as exc:
            raise APIError(f"API request failed: {exc}") from exc

        if response.ok:
            try:
                return response.json()
            except ValueError as exc:
                raise APIError("The API returned a non-JSON response.") from exc
        detail: Any
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except ValueError:
            detail = response.text.strip() or "Unknown API error"
        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )
        raise APIError(f"API error ({response.status_code}): {detail}")

    def check_health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health", timeout=3)

    def get_hardware(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/hardware", timeout=20)

    def get_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/models", timeout=10)

    def estimate(self, config: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/estimate", json=config, timeout=60)

    def run_benchmark(self, config: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        payload = {**config, "benchmark": options}
        return self._request("POST", "/api/v1/benchmark/run", json=payload, timeout=120)
