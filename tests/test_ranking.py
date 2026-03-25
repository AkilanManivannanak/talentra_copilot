from datetime import datetime, timezone

from app.core.config import Settings
from app.models.schemas import CandidateRecord, Requirement, RoleRecord
from app.services.ranking import RankingService


class FakeMetadata:
    def __init__(self) -> None:
        self.role = RoleRecord(
            id="role1",
            title="AI Engineer",
            description="Build RAG systems",
            requirements=[
                Requirement(id="r1", text="Python and FastAPI", weight=1.0),
                Requirement(id="r2", text="RAG systems", weight=1.0),
            ],
            document_ids=[],
            created_at=datetime.now(timezone.utc),
        )
        self.candidates = [
            CandidateRecord(id="c1", name="Alice", document_ids=[], created_at=datetime.now(timezone.utc)),
            CandidateRecord(id="c2", name="Bob", document_ids=[], created_at=datetime.now(timezone.utc)),
        ]

    def get_role(self, role_id: str):
        assert role_id == "role1"
        return self.role

    def list_candidates(self):
        return self.candidates


class FakeVectorStore:
    def search(self, *, query: str, k: int, filters: dict[str, str], fetch_k: int = 30):
        candidate_id = filters["entity_id"]
        if candidate_id == "c1":
            score = 0.8 if "Python" in query else 0.75
            return [{"content": "Built FastAPI RAG service.", "metadata": {"document_id": "d1", "filename": "alice.pdf", "entity_id": "c1", "entity_name": "Alice"}, "score": score}]
        score = 0.25 if "Python" in query else 0.2
        return [{"content": "General software engineer profile.", "metadata": {"document_id": "d2", "filename": "bob.pdf", "entity_id": "c2", "entity_name": "Bob"}, "score": score}]


class FakeSummary:
    def candidate_summary(self, *, role, candidate_name, assessments):
        return f"Summary for {candidate_name}"


def test_ranking_service_orders_best_candidate_first() -> None:
    settings = Settings(openai_api_key="", match_threshold=0.42)
    service = RankingService(
        settings=settings,
        metadata=FakeMetadata(),
        vectorstore=FakeVectorStore(),
        summary_service=FakeSummary(),
    )
    response = service.evaluate_role(role_id="role1", candidate_ids=[], top_k_per_requirement=1)
    assert response.candidates[0].candidate_name == "Alice"
    assert response.candidates[0].matched_requirements == 2
    assert response.candidates[1].matched_requirements == 0
