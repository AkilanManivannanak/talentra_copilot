"""The bias audit must measure the system, not guess at demographics."""
import tempfile
from pathlib import Path

from app.models.schemas import CandidateEvaluation, Requirement, RequirementAssessment
from app.services.bias_audit import (
    CounterfactualBiasAuditor,
    redact_employer,
    redact_name,
    redact_school,
)
from app.services.vectorstore import VectorStoreService


def test_no_demographic_inference_remains_in_the_codebase():
    """Guard against the previous implementation coming back.

    It classified candidates by matching first names against a 29-name gender lookup and
    flagged group-mean gaps computed over as few as one candidate per group.
    """
    import app.agents.bias_auditor as agent_module
    import app.services.bias_audit as service_module

    for module in (service_module, agent_module):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for banned in ("_typically_female", "_typically_male", "gender_scores", "infer_gender"):
            assert banned not in source, f"{banned} reintroduced in {module.__name__}"


def test_redactors_remove_the_signal_they_name():
    text = "Carnegie Mellon University graduate. Worked at Google. Priya Raman led the team."
    assert "carnegie mellon" not in redact_school(text).lower()
    assert "google" not in redact_employer(text).lower()
    assert "priya" not in redact_name(text, "Priya Raman").lower()


def test_redactors_leave_unrelated_evidence_intact():
    text = "Built FastAPI services at Google and shipped a retrieval system."
    redacted = redact_employer(text)
    assert "FastAPI" in redacted and "retrieval system" in redacted


def _evaluation(candidate_id: str, name: str, score: float, requirement: Requirement):
    return CandidateEvaluation(
        candidate_id=candidate_id, candidate_name=name, overall_score=score,
        matched_requirements=1, missing_requirements=0, summary="",
        assessments=[RequirementAssessment(requirement=requirement, score=score, covered=True, evidence=[])],
    )


def test_audit_reports_a_delta_when_a_brand_carries_the_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStoreService(
            embedding_model="local-lexical",
            vectorstore_path=str(Path(tmpdir) / "index.json"),
            openai_api_key="",
        )
        store.add_document(
            text=("Priya Raman, Carnegie Mellon University. Engineer at Google working on "
                  "distributed retrieval infrastructure and evaluation pipelines for search."),
            metadata_base={"document_id": "d1", "entity_type": "candidate", "entity_id": "c1",
                           "entity_name": "Priya Raman", "filename": "p.txt"},
        )
        requirement = Requirement(id="r1", text="Experience at Google with retrieval infrastructure", weight=1.0)
        auditor = CounterfactualBiasAuditor(vectorstore=store, match_threshold=0.28)
        report = auditor.audit(
            requirements=[requirement],
            evaluations=[_evaluation("c1", "Priya Raman", 0.7, requirement)],
        )

        assert report.method == "counterfactual-redaction"
        assert report.candidates_audited == 1
        assert report.severity in {"none", "low", "medium", "high"}
        # Every delta must name an observable signal — never an inferred attribute.
        assert all(delta.signal in {"school", "employer", "name", "all"} for delta in report.deltas)


def test_uncovered_requirements_are_flagged():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStoreService(
            embedding_model="local-lexical",
            vectorstore_path=str(Path(tmpdir) / "index.json"),
            openai_api_key="",
        )
        store.add_document(
            text="Engineer with broad backend and infrastructure delivery experience across teams.",
            metadata_base={"document_id": "d1", "entity_type": "candidate", "entity_id": "c1",
                           "entity_name": "Sam", "filename": "s.txt"},
        )
        covered = Requirement(id="r1", text="Backend infrastructure delivery experience", weight=1.0)
        uncovered = Requirement(id="r2", text="Formal verification of distributed consensus protocols", weight=1.0)
        auditor = CounterfactualBiasAuditor(vectorstore=store, match_threshold=0.28)
        evaluation = CandidateEvaluation(
            candidate_id="c1", candidate_name="Sam", overall_score=0.5,
            matched_requirements=1, missing_requirements=1, summary="",
            assessments=[
                RequirementAssessment(requirement=covered, score=0.6, covered=True, evidence=[]),
                RequirementAssessment(requirement=uncovered, score=0.0, covered=False, evidence=[]),
            ],
        )
        report = auditor.audit(requirements=[covered, uncovered], evaluations=[evaluation])
        assert uncovered.text in report.uncovered_requirements
        assert any("matched no candidate" in flag for flag in report.flags)
