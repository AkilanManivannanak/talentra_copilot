from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# --- Core primitives ---

class Requirement(BaseModel):
    text: str
    weight: float = 1.0


class Evidence(BaseModel):
    document_id: str
    filename: str
    entity_id: str
    entity_name: str
    snippet: str
    score: float


class RequirementAssessment(BaseModel):
    requirement: Requirement
    score: float
    covered: bool
    evidence: list[Evidence] = []


# --- Records ---

class RoleRecord(BaseModel):
    id: str
    title: str
    description: str
    requirements: list[Requirement] = []
    document_ids: list[str] = []
    created_at: datetime


class CandidateRecord(BaseModel):
    id: str
    name: str
    document_ids: list[str] = []
    stage: str = "Applied"
    shortlisted: bool = False
    created_at: datetime


class DocumentRecord(BaseModel):
    id: str
    filename: str
    entity_type: str
    entity_id: str
    entity_name: str
    chunk_count: int
    uploaded_at: datetime


class RecruiterNote(BaseModel):
    id: str
    candidate_id: str
    text: str
    created_at: datetime


# --- Evaluation ---

class CandidateEvaluation(BaseModel):
    candidate_id: str
    candidate_name: str
    overall_score: float
    matched_requirements: int
    missing_requirements: int
    summary: str
    assessments: list[RequirementAssessment] = []


class EvaluateRoleRequest(BaseModel):
    candidate_ids: list[str] | None = None
    top_k_per_requirement: int = 2


class EvaluateRoleResponse(BaseModel):
    role: RoleRecord
    candidates: list[CandidateEvaluation] = []


# --- Roles ---

class RoleCreateRequest(BaseModel):
    title: str
    description: str


class RoleListResponse(BaseModel):
    roles: list[RoleRecord] = []


# --- Candidates ---

class CandidateUploadItem(BaseModel):
    id: str
    name: str
    filename: str
    chunk_count: int


class CandidateUploadFailure(BaseModel):
    filename: str
    error: str


class CandidateUploadResponse(BaseModel):
    candidates: list[CandidateUploadItem] = []
    failed: list[CandidateUploadFailure] = []


class CandidateListResponse(BaseModel):
    candidates: list[CandidateRecord] = []


# --- Copilot ---

class CopilotQueryRequest(BaseModel):
    question: str
    role_id: str
    candidate_ids: list[str] | None = None
    top_k: int = 8


class CopilotAnswerResponse(BaseModel):
    answer: str
    citations: list[Evidence] = []
    reasoning_trace: list[str] = []


# --- Documents ---

class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord] = []


# --- ATS ---

class StageCount(BaseModel):
    stage: str
    count: int


class CandidatePipelineView(BaseModel):
    candidate_id: str
    candidate_name: str
    stage: str
    shortlisted: bool
    notes_count: int
    latest_score: float
    matched_requirements: int
    missing_requirements: int
    summary: str


class ATSDashboardResponse(BaseModel):
    role: RoleRecord
    stage_counts: list[StageCount] = []
    candidates: list[CandidatePipelineView] = []
    shortlist: list[CandidatePipelineView] = []


class CandidateStageUpdateRequest(BaseModel):
    stage: str


class CandidateShortlistRequest(BaseModel):
    shortlisted: bool


class RecruiterNoteCreateRequest(BaseModel):
    text: str


class RecruiterNotesResponse(BaseModel):
    notes: list[RecruiterNote] = []


# --- Ops ---

class OpsMetricsResponse(BaseModel):
    metrics: dict[str, Any] = {}


class OpsBuildInfoResponse(BaseModel):
    build_info: dict[str, Any] = {}


# --- Health ---

class HealthResponse(BaseModel):
    status: str
