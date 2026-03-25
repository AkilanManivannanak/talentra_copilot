from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import OpsBuildInfoResponse, OpsMetricsResponse

router = APIRouter(tags=["ops"])


@router.get("/ops/metrics", response_model=OpsMetricsResponse)
async def ops_metrics(request: Request) -> OpsMetricsResponse:
    metrics_store = request.app.state.metrics
    return OpsMetricsResponse.model_validate(metrics_store.summary())


@router.get("/ops/build", response_model=OpsBuildInfoResponse)
async def ops_build(request: Request) -> OpsBuildInfoResponse:
    settings = request.app.state.settings
    return OpsBuildInfoResponse(
        app_env=settings.app_env,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        external_cost_usd_per_request=0.0,
        features=[
            "local lexical retrieval",
            "requirement-aware ranking",
            "evaluation-aware Q&A routing",
            "ATS-lite workflow",
            "request metrics",
            "Docker + CI",
        ],
        deployment_targets=["docker-compose", "Render", "GitHub Actions CI"],
        notes={
            "external_api_dependency": "disabled by default",
            "cost_model": "local CPU only; no external API spend required",
        },
    )
