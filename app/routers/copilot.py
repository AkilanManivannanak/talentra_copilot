from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.schemas import CopilotAnswerResponse, CopilotQueryRequest
from app.routers.deps import get_services
from app.services.container import ServiceContainer

router = APIRouter(tags=["copilot"])


@router.post("/copilot/query", response_model=CopilotAnswerResponse)
async def copilot_query(
    body: CopilotQueryRequest,
    services: ServiceContainer = Depends(get_services),
) -> CopilotAnswerResponse:
    return services.copilot.answer(
        question=body.question,
        role_id=body.role_id,
        candidate_ids=body.candidate_ids,
        top_k=body.top_k,
    )
