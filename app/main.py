from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.observability import ObservabilityMiddleware, RequestMetricsStore
from app.models.schemas import HealthResponse
from app.routers import ats, candidates, copilot, documents, ops, roles
from app.services.container import ServiceContainer

_metrics_store = RequestMetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json)
    app.state.settings = settings
    app.state.metrics = _metrics_store
    app.state.services = ServiceContainer.from_settings(settings)
    yield


app = FastAPI(title="Talentra Copilot API", lifespan=lifespan)

# Instrument BEFORE routes — middleware hook registers here
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
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
app.include_router(ops.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


# Manually register /metrics route — bypasses expose() timing issues entirely
@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
