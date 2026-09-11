from pathlib import Path

from app.core.config import Settings
from app.models.schemas import Requirement
from app.services.ats import ATSService
from app.services.metadata_store import MetadataStore
from app.services.ranking import RankingService


class FakeVectorStore:
    """Implements the grouped-search contract RankingService now uses: one corpus-wide
    retrieval per requirement, results bucketed by entity_id."""

    def _hits(self, query: str):
        alice_score = 0.8 if "Python" in query else 0.75
        bob_score = 0.25 if "Python" in query else 0.2
        return {
            "c1": [{"content": "Built FastAPI RAG service.", "snippet": "Built FastAPI RAG service.",
                    "metadata": {"document_id": "d1", "filename": "alice.pdf", "entity_id": "c1",
                                 "entity_name": "Alice", "chunk_id": "d1:0"}, "score": alice_score}],
            "c2": [{"content": "General software engineer profile.", "snippet": "General software engineer profile.",
                    "metadata": {"document_id": "d2", "filename": "bob.pdf", "entity_id": "c2",
                                 "entity_name": "Bob", "chunk_id": "d2:0"}, "score": bob_score}],
        }

    def search_grouped(self, *, query: str, per_entity_k: int = 2, filters=None, fetch_k: int = 200):
        return {cid: hits[:per_entity_k] for cid, hits in self._hits(query).items()}

    def search(self, *, query: str, k: int = 5, filters=None, fetch_k: int = 30):
        entity_id = (filters or {}).get("entity_id")
        grouped = self._hits(query)
        if entity_id:
            return grouped.get(entity_id, [])[:k]
        return [hit for hits in grouped.values() for hit in hits][:k]


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
    metadata.create_or_get_candidate("Bob")
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
