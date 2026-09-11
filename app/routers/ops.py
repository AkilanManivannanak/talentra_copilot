from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.models.schemas import OpsBuildInfoResponse, OpsMetricsResponse
from app.routers.deps import get_services
from app.services.container import ServiceContainer

router = APIRouter(tags=["ops"])


@router.get("/ops/metrics", response_model=OpsMetricsResponse)
async def ops_metrics(request: Request) -> OpsMetricsResponse:
    """Rolling in-process latency window.

    Previously this returned `{"metrics": {}}`: the response model declared a single
    `metrics` field, and pydantic v2's `extra="ignore"` silently dropped every key of
    the summary dict. The model now mirrors the summary's real shape, and
    `tests/test_ops.py` asserts the payload is populated.
    """
    return OpsMetricsResponse.model_validate(request.app.state.metrics.summary())


@router.get("/ops/build", response_model=OpsBuildInfoResponse)
async def ops_build(
    request: Request,
    services: ServiceContainer = Depends(get_services),
) -> OpsBuildInfoResponse:
    """What this process can actually do right now — resolved at runtime, not asserted.

    Every optional capability is probed rather than assumed, so the endpoint is the
    authoritative answer to "is the dense/agentic/NER path really active in this
    deployment?" and the README cannot drift away from it.
    """
    from app.graph.hiring_graph import langgraph_available
    from app.preprocessing.pii import presidio_available
    from app.preprocessing.skills import spacy_available

    settings = request.app.state.settings
    retriever = services.retriever
    dense_ready = services.embedder.available

    features = [
        f"retrieval: {retriever.mode}"
        + (f" (requested: {retriever.requested_mode}, degraded)"
           if retriever.mode != retriever.requested_mode else ""),
        "requirement-aware ranking with per-requirement evidence",
        "evaluation-aware Q&A routing",
        "sentence-level answer groundedness scoring",
        "counterfactual bias audit (no demographic inference)",
        "ATS-lite workflow with human-in-the-loop approval",
        "request metrics + Prometheus scrape",
    ]
    if retriever.rerank_active:
        features.append("cross-encoder reranking")

    return OpsBuildInfoResponse(
        app_env=settings.app_env,
        chat_model=settings.chat_model,
        retrieval_mode=retriever.mode,
        embedding_model=settings.embedding_model if dense_ready else "none (lexical only)",
        dense_backend_available=dense_ready,
        rerank_enabled=retriever.rerank_active,
        spacy_available=spacy_available(),
        presidio_available=presidio_available(),
        langgraph_available=langgraph_available(),
        external_cost_usd_per_request=0.0,
        features=features,
        deployment_targets=["docker-compose", "Render", "GitHub Actions CI"],
        notes={
            "external_api_dependency": "none; all models run locally on CPU",
            "cost_model": "no external API spend at any retrieval mode",
            "dense_backend_note": (
                services.embedder.load_error
                if (settings.retrieval_mode != "lexical" and not dense_ready)
                else "installed" if dense_ready else "not requested"
            ),
        },
    )
