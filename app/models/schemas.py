from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Core primitives ---

class Requirement(BaseModel):
    id: str = ""
    text: str
    weight: float = 1.0
    must_have: bool = False


class RetrievalProvenance(BaseModel):
    """Where a piece of evidence came from — populated by the hybrid retriever."""
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    source: Literal["lexical", "dense", "both"] = "lexical"


class Evidence(BaseModel):
    document_id: str
    filename: str
    entity_id: str
    entity_name: str
    snippet: str
    score: float
    chunk_id: str = ""
    provenance: RetrievalProvenance | None = None


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
    skills: list[str] = []
    pii_types_redacted: list[str] = []
    total_years_experience: float = 0.0
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
    skills: list[str] = []
    pii_types_redacted: list[str] = []


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


class SentenceGrounding(BaseModel):
    """One sentence of an answer, and the citation that supports it."""
    sentence: str
    supported: bool
    best_citation_index: int | None = None
    support_score: float = 0.0


class FaithfulnessReport(BaseModel):
    """Sentence-level groundedness of a generated answer against its own citations."""
    sentences_total: int
    sentences_supported: int
    faithfulness: float = Field(description="supported / total, 0.0-1.0")
    threshold: float
    unsupported: list[str] = []
    detail: list[SentenceGrounding] = []


class CopilotAnswerResponse(BaseModel):
    answer: str
    citations: list[Evidence] = []
    reasoning_trace: list[str] = []
    faithfulness: FaithfulnessReport | None = None


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


# --- Agentic path (LangGraph) ---

class ScreenResult(BaseModel):
    candidate_id: str
    candidate_name: str
    screen_pass: bool
    fail_reasons: list[str] = []
    notes: str = ""


class CounterfactualDelta(BaseModel):
    """Score change when an identity signal is removed from the resume."""
    candidate_id: str
    candidate_name: str
    signal: Literal["school", "employer", "name", "all"]
    baseline_score: float
    redacted_score: float
    delta: float


class BiasAuditReport(BaseModel):
    """Counterfactual + coverage bias audit. No demographic inference is performed."""
    method: str = "counterfactual-redaction"
    candidates_audited: int
    deltas: list[CounterfactualDelta] = []
    flags: list[str] = []
    severity: Literal["none", "low", "medium", "high"] = "none"
    recommendation: str = ""
    max_abs_delta: float = 0.0
    uncovered_requirements: list[str] = []


class AgenticEvaluateRequest(BaseModel):
    candidate_ids: list[str] | None = None
    question: str | None = None
    top_k_per_requirement: int = 3
    run_bias_audit: bool = True
    ats_action: dict[str, Any] | None = None


class AgenticEvaluateResponse(BaseModel):
    role_id: str
    role_title: str
    run_id: str
    route_taken: str
    nodes_executed: list[str] = []
    screening: list[ScreenResult] = []
    evaluation: EvaluateRoleResponse | None = None
    answer: CopilotAnswerResponse | None = None
    bias_audit: BiasAuditReport | None = None
    interrupted_before: str | None = None
    pending_ats_action: dict[str, Any] | None = None
    errors: list[str] = []


class AgenticResumeRequest(BaseModel):
    run_id: str
    approve: bool


class AgenticResumeResponse(BaseModel):
    run_id: str
    approved: bool
    ats_committed: bool
    applied_action: dict[str, Any] | None = None
    message: str


class InterviewQuestion(BaseModel):
    type: Literal["behavioral", "technical", "mixed"]
    skill_focus: str = ""
    question: str


class InterviewKitResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    role_title: str
    questions: list[InterviewQuestion] = []
    generator: str = "template"


# --- Ops ---

class RouteMetrics(BaseModel):
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
    routes: list[RouteMetrics] = []


class OpsBuildInfoResponse(BaseModel):
    app_env: str
    chat_model: str
    retrieval_mode: str
    embedding_model: str
    dense_backend_available: bool
    rerank_enabled: bool
    spacy_available: bool
    presidio_available: bool
    langgraph_available: bool
    external_cost_usd_per_request: float
    features: list[str] = []
    deployment_targets: list[str] = []
    notes: dict[str, Any] = {}


# --- Health ---

class HealthResponse(BaseModel):
    status: str
