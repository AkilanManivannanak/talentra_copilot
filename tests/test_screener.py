from datetime import UTC, datetime

from app.agents.screener import ScreenerAgent
from app.models.schemas import CandidateRecord, Requirement


def candidate(cid: str, name: str, skills=None) -> CandidateRecord:
    return CandidateRecord(id=cid, name=name, document_ids=[], skills=skills or [],
                           created_at=datetime.now(UTC))


class FakeStore:
    """Implements the corpus-wide grouped contract the screener uses."""

    def __init__(self, hits_by_candidate):
        self._hits = hits_by_candidate

    def search_grouped(self, *, query, per_entity_k=2, filters=None, fetch_k=200):
        return {cid: hits[:per_entity_k] for cid, hits in self._hits.items() if hits}

    def search(self, *, query, k=2, filters=None, fetch_k=20):
        entity_id = (filters or {}).get("entity_id")
        if entity_id:
            return self._hits.get(entity_id, [])
        return [hit for hits in self._hits.values() for hit in hits]


def hit(content: str, score: float):
    return {"content": content, "snippet": content, "score": score,
            "metadata": {"document_id": "d", "filename": "f", "entity_id": "c",
                         "entity_name": "n", "chunk_id": "d:0"}}


def test_everyone_passes_when_there_are_no_must_haves():
    results = ScreenerAgent().screen(
        role_requirements=[Requirement(id="r1", text="Nice to have Docker", must_have=False)],
        candidates=[candidate("c1", "A"), candidate("c2", "B")],
        vectorstore=FakeStore({}),
    )
    assert all(r.screen_pass for r in results)
    assert all("No must-have" in r.notes for r in results)


def test_candidate_without_evidence_fails_the_must_have():
    requirement = Requirement(id="r1", text="Required: retrieval-augmented generation in production",
                              must_have=True)
    store = FakeStore({"c1": [hit("Built retrieval-augmented generation pipelines in production", 0.8)],
                       "c2": [hit("Frontend work in TypeScript and CSS", 0.05)]})
    results = {r.candidate_id: r for r in ScreenerAgent().screen(
        role_requirements=[requirement],
        candidates=[candidate("c1", "A"), candidate("c2", "B")],
        vectorstore=store,
    )}
    assert results["c1"].screen_pass
    assert not results["c2"].screen_pass
    assert requirement.text in results["c2"].fail_reasons


def test_a_generic_skill_token_cannot_clear_a_must_have_alone():
    """Regression: 'python' in both the requirement and the skill list used to satisfy
    'Must have 3+ years of professional Python experience' with no evidence at all."""
    requirement = Requirement(id="r1", text="Must have 3+ years of professional Python experience",
                              must_have=True)
    store = FakeStore({
        "c1": [hit("Class projects and coursework only.", 0.05)],
        "c2": [hit("Eight years of professional Python engineering in production.", 0.9)],
    })
    results = {r.candidate_id: r for r in ScreenerAgent().screen(
        role_requirements=[requirement],
        candidates=[candidate("c1", "A", skills=["python"]), candidate("c2", "B", skills=["python"])],
        vectorstore=store,
    )}
    assert not results["c1"].screen_pass
    assert results["c2"].screen_pass


def test_a_distinctive_skill_does_clear_a_must_have():
    requirement = Requirement(id="r1", text="Required: experience with LangGraph orchestration",
                              must_have=True)
    results = ScreenerAgent().screen(
        role_requirements=[requirement],
        candidates=[candidate("c1", "A", skills=["langgraph"])],
        vectorstore=FakeStore({}),
    )
    assert results[0].screen_pass


def test_a_candidate_far_below_the_pool_fails_even_with_a_nonzero_score():
    """Regression: a resume saying 'no retrieval, deployment, or production ML systems'
    cleared a RAG must-have on the shared word 'production'. Bag-of-words cannot see
    negation, so the screener judges relative to what the rest of the pool produced."""
    requirement = Requirement(id="r1", text="Required: retrieval-augmented generation systems in production",
                              must_have=True)
    store = FakeStore({
        "c1": [hit("Worked on production support rotas and incident response.", 0.30)],
        "c2": [hit("Shipped retrieval-augmented generation systems to production.", 0.95)],
    })
    results = {r.candidate_id: r for r in ScreenerAgent().screen(
        role_requirements=[requirement],
        candidates=[candidate("c1", "A"), candidate("c2", "B")],
        vectorstore=store,
    )}
    assert not results["c1"].screen_pass
    assert results["c2"].screen_pass


def test_screener_passes_when_it_has_no_retrieval_signal_at_all():
    """A false negative eliminates someone entirely, so absence of information is not
    treated as evidence of absence."""
    requirement = Requirement(id="r1", text="Required: Kubernetes operator development", must_have=True)
    results = ScreenerAgent().screen(
        role_requirements=[requirement],
        candidates=[candidate("c1", "A")],
        vectorstore=FakeStore({}),
    )
    assert results[0].screen_pass
