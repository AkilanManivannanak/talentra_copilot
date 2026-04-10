from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.schemas import (
    ATSDashboardResponse,
    CandidateRecord,
    CandidateShortlistRequest,
    CandidateStageUpdateRequest,
    RecruiterNote,
    RecruiterNoteCreateRequest,
    RecruiterNotesResponse,
)
from app.routers.deps import get_services
from app.services.container import ServiceContainer

router = APIRouter(tags=["ats"])


@router.get("/ats/roles/{role_id}/dashboard", response_model=ATSDashboardResponse)
async def ats_dashboard(
    role_id: str,
    candidate_ids: list[str] | None = None,
    services: ServiceContainer = Depends(get_services),
) -> ATSDashboardResponse:
    return services.ats.dashboard(role_id=role_id, candidate_ids=candidate_ids or [])


@router.post("/ats/candidates/{candidate_id}/stage", response_model=CandidateRecord)
async def update_candidate_stage(
    candidate_id: str,
    body: CandidateStageUpdateRequest,
    services: ServiceContainer = Depends(get_services),
) -> CandidateRecord:
    return services.ats.update_stage(candidate_id=candidate_id, stage=body.stage)


@router.post("/ats/candidates/{candidate_id}/shortlist", response_model=CandidateRecord)
async def update_candidate_shortlist(
    candidate_id: str,
    body: CandidateShortlistRequest,
    services: ServiceContainer = Depends(get_services),
) -> CandidateRecord:
    return services.ats.update_shortlist(candidate_id=candidate_id, shortlisted=body.shortlisted)


@router.post("/ats/candidates/{candidate_id}/notes", response_model=RecruiterNote)
async def create_candidate_note(
    candidate_id: str,
    body: RecruiterNoteCreateRequest,
    services: ServiceContainer = Depends(get_services),
) -> RecruiterNote:
    return services.ats.add_note(candidate_id=candidate_id, text=body.text)


@router.get("/ats/candidates/{candidate_id}/notes", response_model=RecruiterNotesResponse)
async def list_candidate_notes(
    candidate_id: str,
    services: ServiceContainer = Depends(get_services),
) -> RecruiterNotesResponse:
    return RecruiterNotesResponse(notes=services.ats.list_notes(candidate_id=candidate_id))
