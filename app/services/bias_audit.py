"""
Counterfactual bias auditing.

The previous implementation inferred gender from a hardcoded list of ~29 first names
and flagged a group-mean gap. That is exactly the mechanism responsible-AI tooling
exists to prevent: it fabricates a protected attribute the system was never given, and
it fails hardest on names outside the list it was written from. It has been removed and
is not coming back.

What replaces it measures something the system can actually observe: how much of a
candidate's score depends on identity signals rather than demonstrated evidence.

For each candidate we re-score the resume with a signal redacted and report the delta:

    delta_school   = score(resume) - score(resume with school names removed)
    delta_employer = score(resume) - score(resume with employer brands removed)
    delta_name     = score(resume) - score(resume with the person's name removed)

A large positive delta means the pipeline is rewarding a brand rather than the work.
A large negative delta means redaction destroyed real evidence, which is a retrieval
bug worth knowing about. Either way the number is derived from the system's own
behaviour, on the actual documents, with no demographic inference anywhere.

The audit also reports requirements no candidate covers, which is usually an extraction
failure rather than a pool failure and is the most common source of silent unfairness.
"""
from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from app.models.schemas import BiasAuditReport, CandidateEvaluation, CounterfactualDelta, Requirement
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

# Brand tokens whose presence in a resume is an identity signal rather than evidence of skill.
_PRESTIGE_SCHOOLS = [
    "mit", "massachusetts institute of technology", "stanford", "harvard", "oxford", "cambridge",
    "princeton", "yale", "columbia", "caltech", "california institute of technology",
    "carnegie mellon", "cmu", "berkeley", "uc berkeley", "cornell", "iit", "eth zurich",
]
_EMPLOYER_BRANDS = [
    "google", "alphabet", "meta", "facebook", "amazon", "aws", "apple", "microsoft", "netflix",
    "openai", "anthropic", "nvidia", "tesla", "waymo", "deepmind", "stripe", "uber", "airbnb",
    "goldman sachs", "jpmorgan", "mckinsey", "palantir", "databricks",
]

_SCHOOL_PATTERN = re.compile(
    r"\b(?:university of [a-z][a-z\- ]{2,30}"
    r"|[A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)* (?:University|College|Institute of Technology|Polytechnic))\b",
    re.IGNORECASE,
)


def _redact_terms(text: str, terms: Sequence[str], placeholder: str) -> str:
    redacted = text
    for term in sorted(terms, key=len, reverse=True):
        redacted = re.sub(rf"\b{re.escape(term)}\b", placeholder, redacted, flags=re.IGNORECASE)
    return redacted


def redact_school(text: str) -> str:
    text = _SCHOOL_PATTERN.sub("<SCHOOL>", text)
    return _redact_terms(text, _PRESTIGE_SCHOOLS, "<SCHOOL>")


def redact_employer(text: str) -> str:
    return _redact_terms(text, _EMPLOYER_BRANDS, "<EMPLOYER>")


def redact_name(text: str, candidate_name: str) -> str:
    parts = [part for part in re.split(r"\s+", candidate_name.strip()) if len(part) > 2]
    return _redact_terms(text, parts, "<NAME>") if parts else text


class CounterfactualBiasAuditor:
    """Re-scores redacted variants of each resume in a throwaway index and reports deltas."""

    # Below this, a delta is indistinguishable from scoring noise at this corpus size.
    NOISE_FLOOR = 0.02
    # Above this, a single identity token is moving the score enough to change a ranking.
    FLAG_THRESHOLD = 0.08

    def __init__(self, *, vectorstore: VectorStoreService, match_threshold: float) -> None:
        self._vectorstore = vectorstore
        self._match_threshold = match_threshold

    def audit(
        self,
        *,
        requirements: Sequence[Requirement],
        evaluations: Sequence[CandidateEvaluation],
        signals: Sequence[str] = ("school", "employer", "name"),
    ) -> BiasAuditReport:
        deltas: list[CounterfactualDelta] = []
        flags: list[str] = []

        for evaluation in evaluations:
            text = self._vectorstore.text_for_entity(evaluation.candidate_id)
            if not text.strip():
                continue
            baseline = self._score_text(text, requirements)
            for signal in signals:
                variant = self._apply(text, signal, evaluation.candidate_name)
                if variant == text:
                    continue  # signal not present in this resume; nothing to measure
                redacted_score = self._score_text(variant, requirements)
                delta = round(baseline - redacted_score, 4)
                if abs(delta) < self.NOISE_FLOOR:
                    continue
                deltas.append(
                    CounterfactualDelta(
                        candidate_id=evaluation.candidate_id,
                        candidate_name=evaluation.candidate_name,
                        signal=signal,  # type: ignore[arg-type]
                        baseline_score=round(baseline, 4),
                        redacted_score=round(redacted_score, 4),
                        delta=delta,
                    )
                )

        for item in deltas:
            if item.delta >= self.FLAG_THRESHOLD:
                flags.append(
                    f"{item.candidate_name}: removing {item.signal} signals drops the score by "
                    f"{item.delta:.3f} ({item.baseline_score:.3f} to {item.redacted_score:.3f}). "
                    f"The ranking is partly rewarding a brand name rather than demonstrated work."
                )
            elif item.delta <= -self.FLAG_THRESHOLD:
                flags.append(
                    f"{item.candidate_name}: removing {item.signal} signals *raises* the score by "
                    f"{abs(item.delta):.3f}. Redaction is likely destroying real evidence — "
                    f"check chunking around that section."
                )

        uncovered = self._uncovered_requirements(requirements, evaluations)
        if uncovered:
            flags.append(
                f"{len(uncovered)} requirement(s) matched no candidate at all: "
                f"{'; '.join(uncovered[:3])}. This is usually a requirement-extraction failure, "
                f"not a pool failure, and it silently penalises everyone equally."
            )

        max_abs = max((abs(item.delta) for item in deltas), default=0.0)
        severity = self._severity(flags, max_abs)
        return BiasAuditReport(
            candidates_audited=len(evaluations),
            deltas=deltas,
            flags=flags,
            severity=severity,
            recommendation={
                "none": "No identity signal moved a score beyond the noise floor. Proceed.",
                "low": "One weak signal detected. Note it; no action required before review.",
                "medium": "Identity signals are measurably moving scores. Review the flagged candidates "
                          "against the redacted evidence before advancing anyone.",
                "high": "Identity signals dominate at least one score. Do not advance on this ranking; "
                        "re-run with redaction enabled at ingest.",
            }[severity],
            max_abs_delta=round(max_abs, 4),
            uncovered_requirements=uncovered,
        )

    # -- internals ---------------------------------------------------------

    def _apply(self, text: str, signal: str, candidate_name: str) -> str:
        if signal == "school":
            return redact_school(text)
        if signal == "employer":
            return redact_employer(text)
        if signal == "name":
            return redact_name(text, candidate_name)
        if signal == "all":
            return redact_name(redact_employer(redact_school(text)), candidate_name)
        return text

    def _score_text(self, text: str, requirements: Sequence[Requirement]) -> float:
        """Score one document in isolation using a throwaway index.

        Isolation matters: IDF is computed over the eligible set, so scoring the variant
        inside the shared index would let the original document's terms skew the variant's
        weights. A fresh index per variant keeps the comparison honest.
        """
        if not requirements:
            return 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = VectorStoreService(
                embedding_model="local-lexical",
                vectorstore_path=str(Path(tmpdir) / "index.json"),
                openai_api_key="",
                cache_size=32,
            )
            try:
                probe.add_document(
                    text=text,
                    metadata_base={
                        "document_id": "probe",
                        "entity_type": "candidate",
                        "entity_id": "probe",
                        "entity_name": "probe",
                        "filename": "probe.txt",
                    },
                )
            except ValueError:
                return 0.0

            scores: list[float] = []
            weights: list[float] = []
            for requirement in requirements:
                hits = probe.search(query=requirement.text, k=2, filters={"entity_id": "probe"})
                best = max((hit["score"] for hit in hits), default=0.0)
                scores.append(best * requirement.weight)
                weights.append(requirement.weight)
            total_weight = sum(weights) or 1.0
            return sum(scores) / total_weight

    def _uncovered_requirements(
        self,
        requirements: Sequence[Requirement],
        evaluations: Sequence[CandidateEvaluation],
    ) -> list[str]:
        if not evaluations:
            return []
        covered_texts = {
            assessment.requirement.text
            for evaluation in evaluations
            for assessment in evaluation.assessments
            if assessment.covered
        }
        return [req.text for req in requirements if req.text not in covered_texts]

    def _severity(self, flags: list[str], max_abs: float) -> str:
        if not flags:
            return "none"
        if max_abs >= 0.20 or len(flags) >= 4:
            return "high"
        if max_abs >= self.FLAG_THRESHOLD or len(flags) >= 2:
            return "medium"
        return "low"
