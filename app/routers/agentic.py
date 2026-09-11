from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.schemas import (
    AgenticEvaluateRequest,
    AgenticEvaluateResponse,
    AgenticResumeRequest,
    AgenticResumeResponse,
    InterviewKitResponse,
)
from app.routers.deps import get_services
from app.services.container import ServiceContainer

router = APIRouter(tags=["agentic"])


@router.post("/roles/{role_id}/evaluate/agentic", response_model=AgenticEvaluateResponse)
async def evaluate_agentic(
    role_id: str,
    body: AgenticEvaluateRequest,
    request: Request,
    services: ServiceContainer = Depends(get_services),
) -> AgenticEvaluateResponse:
    """Run the full graph: screen -> evaluate -> bias audit -> route -> answer -> [ATS interrupt].

    When `ats_action` is supplied the run stops *before* the write and returns
    `interrupted_before: "ats_update"`. Call `/agentic/runs/{run_id}/resume` to approve.
    """
    runner = getattr(request.app.state, "agentic", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="Agentic path is disabled (AGENTIC_ENABLED=false).")
    try:
        services.metadata.get_role(role_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown role_id: {role_id}") from exc

    if not services.metadata.list_candidates():
        raise HTTPException(status_code=400, detail="No candidates have been uploaded yet.")

    return runner.run(role_id=role_id, request=body)


@router.post("/agentic/runs/{run_id}/resume", response_model=AgenticResumeResponse)
async def resume_agentic_run(
    run_id: str,
    body: AgenticResumeRequest,
    request: Request,
) -> AgenticResumeResponse:
    """Human-in-the-loop approval gate. Nothing is written to the ATS until this returns."""
    runner = getattr(request.app.state, "agentic", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="Agentic path is disabled (AGENTIC_ENABLED=false).")
    return runner.resume(run_id=run_id, approve=body.approve)


@router.post("/roles/{role_id}/candidates/{candidate_id}/interview-kit", response_model=InterviewKitResponse)
async def interview_kit(
    role_id: str,
    candidate_id: str,
    num_questions: int = 8,
    services: ServiceContainer = Depends(get_services),
) -> InterviewKitResponse:
    """Interview questions targeted at this candidate's strongest retrieved evidence."""
    from app.agents.interviewer import InterviewerAgent

    try:
        role = services.metadata.get_role(role_id)
        candidate = services.metadata.get_candidate(candidate_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    evaluation = services.ranking.evaluate_role(role_id=role_id, candidate_ids=[candidate_id])
    top_evidence = []
    if evaluation.candidates:
        for assessment in sorted(evaluation.candidates[0].assessments, key=lambda a: a.score, reverse=True)[:3]:
            top_evidence.extend(assessment.evidence[:1])

    questions = InterviewerAgent().generate_questions(
        role_title=role.title,
        requirements=role.requirements,
        candidate_skills=candidate.skills,
        top_evidence=top_evidence,
        num_questions=max(1, min(num_questions, 20)),
    )
    return InterviewKitResponse(
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        role_title=role.title,
        questions=questions,
        generator="template",
    )
