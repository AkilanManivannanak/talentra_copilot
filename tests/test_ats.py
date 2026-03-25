from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.models.schemas import Requirement, RoleRecord
from app.services.ats import ATSService
from app.services.metadata_store import MetadataStore
from app.services.ranking import RankingService


class FakeVectorStore:
    def search(self, *, query: str, k: int, filters: dict[str, str], fetch_k: int = 30):
        candidate_id = filters["entity_id"]
        if candidate_id == "c1":
            score = 0.8 if "Python" in query else 0.75
            return [{"content": "Built FastAPI RAG service.", "snippet": "Built FastAPI RAG service.", "metadata": {"document_id": "d1", "filename": "alice.pdf", "entity_id": "c1", "entity_name": "Alice"}, "score": score}]
        score = 0.25 if "Python" in query else 0.2
        return [{"content": "General software engineer profile.", "snippet": "General software engineer profile.", "metadata": {"document_id": "d2", "filename": "bob.pdf", "entity_id": "c2", "entity_name": "Bob"}, "score": score}]


class FakeSummary:
    def candidate_summary(self, *, role, candidate_name, assessments):
        return f"Summary for {candidate_name}"


def test_ats_dashboard_includes_stage_and_notes(tmp_path: Path) -> None:
    metadata = MetadataStore(tmp_path)
    role = metadata.create_role(
        title="AI Engineer",
        description="Build RAG systems",
        requirements=[
            Requirement(id="r1", text="Python and FastAPI", weight=1.0),
            Requirement(id="r2", text="RAG systems", weight=1.0),
        ],
    )
    alice = metadata.create_or_get_candidate("Alice")
    bob = metadata.create_or_get_candidate("Bob")
    metadata.update_candidate_stage(alice.id, "Interview")
    metadata.update_candidate_shortlist(alice.id, True)
    metadata.add_note(alice.id, "Strong backend depth.")

    settings = Settings(openai_api_key="", match_threshold=0.42, data_dir=str(tmp_path), vectorstore_path=str(tmp_path / 'index.json'))
    ranking = RankingService(settings=settings, metadata=metadata, vectorstore=FakeVectorStore(), summary_service=FakeSummary())
    ats = ATSService(metadata=metadata, ranking=ranking)

    dashboard = ats.dashboard(role_id=role.id)
    assert dashboard.candidates[0].candidate_name == "Alice"
    assert dashboard.candidates[0].stage == "Interview"
    assert dashboard.candidates[0].shortlisted is True
    assert dashboard.candidates[0].notes_count == 1
    assert any(item.stage == "Interview" and item.count == 1 for item in dashboard.stage_counts)
