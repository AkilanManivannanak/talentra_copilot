from datetime import datetime, timezone

from app.models.schemas import (
    CandidateEvaluation,
    CandidateRecord,
    EvaluateRoleResponse,
    Evidence,
    Requirement,
    RequirementAssessment,
    RoleRecord,
)
from app.services.copilot import CopilotService
from app.services.summary import SummaryService


class FakeMetadata:
    def __init__(self) -> None:
        self.role = RoleRecord(
            id="role1",
            title="ML Intern",
            description="Need machine learning and Python",
            requirements=[
                Requirement(id="r1", text="Python", weight=1.0),
                Requirement(id="r2", text="Machine learning", weight=1.0),
            ],
            document_ids=[],
            created_at=datetime.now(timezone.utc),
        )

    def get_role(self, role_id: str):
        assert role_id == "role1"
        return self.role


class FakeRanking:
    def evaluate_role(self, *, role_id: str, candidate_ids=None, top_k_per_requirement: int = 2):
        role = FakeMetadata().role
        akila = CandidateEvaluation(
            candidate_id="c1",
            candidate_name="Akila Resume 2",
            overall_score=0.44,
            matched_requirements=2,
            missing_requirements=0,
            summary="Akila summary",
            assessments=[
                RequirementAssessment(
                    requirement=role.requirements[0],
                    score=0.50,
                    covered=True,
                    evidence=[Evidence(document_id="d1", filename="akila.pdf", entity_id="c1", entity_name="Akila Resume 2", snippet="Built Python APIs with FastAPI.", score=0.50)],
                ),
                RequirementAssessment(
                    requirement=role.requirements[1],
                    score=0.38,
                    covered=True,
                    evidence=[Evidence(document_id="d1", filename="akila.pdf", entity_id="c1", entity_name="Akila Resume 2", snippet="Machine learning coursework and projects.", score=0.38)],
                ),
            ],
        )
        jaxon = CandidateEvaluation(
            candidate_id="c2",
            candidate_name="Jaxon Poentis Main Resume",
            overall_score=0.22,
            matched_requirements=0,
            missing_requirements=2,
            summary="Jaxon summary",
            assessments=[
                RequirementAssessment(
                    requirement=role.requirements[0],
                    score=0.10,
                    covered=False,
                    evidence=[Evidence(document_id="d2", filename="jaxon.pdf", entity_id="c2", entity_name="Jaxon Poentis Main Resume", snippet="General CS coursework.", score=0.10)],
                ),
                RequirementAssessment(
                    requirement=role.requirements[1],
                    score=0.08,
                    covered=False,
                    evidence=[Evidence(document_id="d2", filename="jaxon.pdf", entity_id="c2", entity_name="Jaxon Poentis Main Resume", snippet="No direct ML evidence.", score=0.08)],
                ),
            ],
        )
        return EvaluateRoleResponse(role=role, candidates=[akila, jaxon])


class FakeVectorStore:
    def search(self, *, query: str, k: int = 5, filters=None, fetch_k: int = 30):
        filters = filters or {}
        entity_id = filters.get("entity_id")
        query = query.lower()
        if "machine learning" in query and entity_id == "c1":
            return [{
                "content": "Akila completed machine learning projects with Python.",
                "snippet": "Akila completed machine learning projects with Python.",
                "metadata": {"document_id": "d1", "filename": "akila.pdf", "entity_id": "c1", "entity_name": "Akila Resume 2"},
                "score": 0.62,
            }]
        if "machine learning" in query and entity_id == "c2":
            return [{
                "content": "Jaxon lists coursework but no strong ML project evidence.",
                "snippet": "Jaxon lists coursework but no strong ML project evidence.",
                "metadata": {"document_id": "d2", "filename": "jaxon.pdf", "entity_id": "c2", "entity_name": "Jaxon Poentis Main Resume"},
                "score": 0.21,
            }]
        return []


def build_service() -> CopilotService:
    return CopilotService(
        metadata=FakeMetadata(),
        vectorstore=FakeVectorStore(),
        summary_service=SummaryService(chat_model="local", openai_api_key=""),
        ranking_service=FakeRanking(),
    )


def test_copilot_uses_evaluation_for_ranking_question() -> None:
    service = build_service()
    response = service.answer(
        question="Which candidate matches the most job requirements?",
        role_id="role1",
        candidate_ids=[],
    )
    assert response.answer.startswith("Based on the evaluation results, Akila Resume 2 is the strongest candidate")


def test_copilot_compares_named_candidates_correctly() -> None:
    service = build_service()
    response = service.answer(
        question="Why is Akila ranked above Jaxon?",
        role_id="role1",
        candidate_ids=[],
    )
    assert "Akila Resume 2 is ranked above Jaxon Poentis Main Resume" in response.answer


def test_copilot_runs_targeted_skill_query() -> None:
    service = build_service()
    response = service.answer(
        question="Who shows the strongest evidence for machine learning?",
        role_id="role1",
        candidate_ids=[],
    )
    assert "Akila Resume 2" in response.answer
    assert any("machine learning" in c.snippet.lower() for c in response.citations)
