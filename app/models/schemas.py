from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


CandidateStage = Literal["Applied", "Screening", "Shortlisted", "Interview", "Final", "Rejected"]


class HealthResponse(BaseModel):
    status: str


class RoleCreateRequest(BaseModel):
    title: str
    description: str


class Requirement(BaseModel):
    id: str
    text: str
    weight: float = Field(default=1.0, ge=0.1, le=3.0)


class RoleRecord(BaseModel):
    id: str
    title: str
    description: str
    requirements: list[Requirement] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class CandidateRecord(BaseModel):
    id: str
    name: str
    document_ids: list[str] = Field(default_factory=list)
    stage: CandidateStage = "Applied"
    shortlisted: bool = False
    created_at: datetime


class RecruiterNote(BaseModel):
    id: str
    candidate_id: str
    text: str
    created_at: datetime


class DocumentRecord(BaseModel):
    id: str
    filename: str
    entity_type: Literal["role", "candidate"]
    entity_id: str
    entity_name: str
    chunk_count: int
    uploaded_at: datetime


class RoleListResponse(BaseModel):
    roles: list[RoleRecord]


class CandidateListResponse(BaseModel):
    candidates: list[CandidateRecord]


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class CandidateUploadItem(BaseModel):
    id: str
    name: str
    filename: str
    chunk_count: int


class CandidateUploadFailure(BaseModel):
    filename: str
    error: str


class CandidateUploadResponse(BaseModel):
    candidates: list[CandidateUploadItem]
    failed: list[CandidateUploadFailure] = Field(default_factory=list)


class Evidence(BaseModel):
    document_id: str
    filename: str
    entity_id: str
    entity_name: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)


class RequirementAssessment(BaseModel):
    requirement: Requirement
    score: float = Field(ge=0.0, le=1.0)
    covered: bool
    evidence: list[Evidence] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    candidate_id: str
    candidate_name: str
    overall_score: float = Field(ge=0.0, le=1.0)
    matched_requirements: int
    missing_requirements: int
    summary: str
    assessments: list[RequirementAssessment]


class EvaluateRoleRequest(BaseModel):
    candidate_ids: list[str] = Field(default_factory=list)
    top_k_per_requirement: int = Field(default=2, ge=1, le=5)


class EvaluateRoleResponse(BaseModel):
    role: RoleRecord
    candidates: list[CandidateEvaluation]


class CopilotQueryRequest(BaseModel):
    question: str
    role_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=3, le=20)


class CopilotAnswerResponse(BaseModel):
    answer: str
    citations: list[Evidence]
    reasoning_trace: list[str]


class CandidateStageUpdateRequest(BaseModel):
    stage: CandidateStage


class CandidateShortlistRequest(BaseModel):
    shortlisted: bool


class RecruiterNoteCreateRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class RecruiterNotesResponse(BaseModel):
    notes: list[RecruiterNote]


class CandidatePipelineView(BaseModel):
    candidate_id: str
    candidate_name: str
    stage: CandidateStage
    shortlisted: bool
    notes_count: int
    latest_score: float | None = None
    matched_requirements: int | None = None
    missing_requirements: int | None = None
    summary: str | None = None


class StageCount(BaseModel):
    stage: CandidateStage
    count: int


class ATSDashboardResponse(BaseModel):
    role: RoleRecord
    stage_counts: list[StageCount]
    candidates: list[CandidatePipelineView]
    shortlist: list[CandidatePipelineView]


class RouteMetric(BaseModel):
    route: str
    requests: int
    errors: int
    avg_ms: float
    p50_ms: float
    p95_ms: float


class OpsMetricsResponse(BaseModel):
    window_size: int
    error_rate: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    routes: list[RouteMetric] = Field(default_factory=list)


class OpsBuildInfoResponse(BaseModel):
    app_env: str
    chat_model: str
    embedding_model: str
    external_cost_usd_per_request: float
    features: list[str] = Field(default_factory=list)
    deployment_targets: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)
