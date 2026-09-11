from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.observability import ObservabilityMiddleware, RequestMetricsStore
from app.models.schemas import HealthResponse
from app.routers import agentic, ats, candidates, copilot, documents, ops, roles
from app.services.agentic import AgenticRunner
from app.services.container import ServiceContainer

_metrics_store = RequestMetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json)
    app.state.settings = settings
    app.state.metrics = _metrics_store
    services = ServiceContainer.from_settings(settings)
    app.state.services = services
    # Compiled once: building the graph binds the live services, and the checkpointer
    # holds parked runs awaiting recruiter approval for the process lifetime.
    app.state.agentic = AgenticRunner(
        services, interrupt_before_ats=settings.agentic_interrupt_before_ats
    ) if settings.agentic_enabled else None

    if settings.prewarm_models and settings.redact_pii_on_ingest:
        # Daemon thread: startup must not block on model construction, and a failure to
        # warm is never fatal — the pipelines build lazily on first use as before.
        threading.Thread(target=_prewarm, name="talentra-prewarm", daemon=True).start()
    yield


def _prewarm() -> None:
    import logging

    from app.preprocessing import warm_models

    try:
        logging.getLogger(__name__).info("Preprocessing models warmed: %s", warm_models())
    except Exception:
        logging.getLogger(__name__).warning("Model prewarm failed; falling back to lazy load.")


app = FastAPI(title="Talentra Copilot API", lifespan=lifespan)

# Instrument BEFORE routes — the middleware hook registers here.
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    inprogress_name="talentra_http_requests_inprogress",
    excluded_handlers=["/health", "/docs", "/openapi.json", "/redoc"],
).instrument(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ObservabilityMiddleware, metrics_store=_metrics_store)

app.include_router(roles.router)
app.include_router(candidates.router)
app.include_router(copilot.router)
app.include_router(documents.router)
app.include_router(ats.router)
app.include_router(agentic.router)
app.include_router(ops.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
