"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.core.config import API_PREFIX, APP_NAME, DISCLAIMER
from app.core.logging import configure_logging
from app.schemas import RootResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        description=(
            "Hardware-aware ML training resource estimator and benchmark service. "
            "Estimates are heuristic planning aids, not guarantees."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Check the API logs for details.",
                "path": request.url.path,
            },
        )

    @app.get("/", response_model=RootResponse, tags=["system"])
    def root() -> RootResponse:
        return RootResponse(
            app=APP_NAME,
            version=__version__,
            docs_url="/docs",
            dashboard_hint="streamlit run dashboard/app.py",
        )

    @app.get("/health", tags=["system"])
    def root_health() -> dict[str, str]:
        return {
            "status": "ok",
            "app": APP_NAME,
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DISCLAIMER,
        }

    app.include_router(router)
    logger.info("%s %s started", APP_NAME, __version__)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
