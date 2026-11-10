"""HTTP API routes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app import __version__
from app.benchmark.runner import (
    BenchmarkExecutionError,
    BenchmarkSafetyError,
    run_benchmark,
)
from app.core.config import API_PREFIX, APP_NAME
from app.hardware.detector import detect_hardware
from app.models.registry import get_models
from app.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ComputeEstimate,
    EstimateResponse,
    HardwareInfo,
    HealthResponse,
    MemoryEstimate,
    ModelConfig,
    ModelInfo,
    RecommendationSet,
    TimeEstimate,
)
from app.services.planner import (
    estimate_plan,
    get_compute,
    get_memory,
    get_recommendations,
    get_time,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix=API_PREFIX)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=APP_NAME,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/hardware", response_model=HardwareInfo, tags=["hardware"])
def hardware() -> HardwareInfo:
    return detect_hardware()


@router.get("/models", response_model=list[ModelInfo], tags=["models"])
def models() -> list[ModelInfo]:
    return get_models()


@router.post("/estimate", response_model=EstimateResponse, tags=["estimation"])
def estimate(config: ModelConfig) -> EstimateResponse:
    return estimate_plan(config)


@router.post("/estimate/compute", response_model=ComputeEstimate, tags=["estimation"])
def estimate_compute_only(config: ModelConfig) -> ComputeEstimate:
    return get_compute(config)


@router.post("/estimate/memory", response_model=MemoryEstimate, tags=["estimation"])
def estimate_memory_only(config: ModelConfig) -> MemoryEstimate:
    return get_memory(config)


@router.post("/estimate/time", response_model=TimeEstimate, tags=["estimation"])
def estimate_time_only(config: ModelConfig) -> TimeEstimate:
    return get_time(config)


@router.post("/recommendations", response_model=RecommendationSet, tags=["recommendations"])
def recommendations(config: ModelConfig) -> RecommendationSet:
    return get_recommendations(config)


@router.post("/benchmark/run", response_model=BenchmarkResponse, tags=["benchmark"])
def benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    try:
        return run_benchmark(request)
    except BenchmarkSafetyError as exc:
        logger.warning("Benchmark rejected by safety checks: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BenchmarkExecutionError as exc:
        logger.error("Benchmark execution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
