"""ScreenerAgent: hard filter on must-have requirements, before the expensive ranking pass."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.models.schemas import CandidateRecord, Requirement, ScreenResult

from .base import BaseAgent

_STOP = {
    "and", "or", "the", "a", "an", "of", "in", "with", "for", "to", "on", "at", "is", "are",
    "experience", "experienced", "knowledge", "skills", "skill", "ability", "strong", "must",
    "have", "required", "require", "requires", "minimum", "least", "years", "year", "plus",
    "demonstrated", "proficient", "proficiency", "familiar", "familiarity", "working",
}


class ScreenerAgent(BaseAgent):
    """Decides who is eligible, using the `must_have` flag the extractor sets at extraction.

    Two design decisions worth stating, because the obvious implementations are both wrong:

    **Screening is pool-relative, not absolute.** An absolute score threshold cannot work
    here. Bag-of-words retrieval cannot see negation: a resume reading "no retrieval,
    deployment, or production ML systems" matches a RAG requirement about as well as one
    reading "shipped retrieval systems to production", because they share the same terms.
    Comparing candidates against each other on one corpus-wide retrieval does separate
    them, because global IDF makes the rare, discriminating terms dominate. So a candidate
    fails a must-have when their evidence is both weak in absolute terms *and* far below
    the best evidence anyone in the pool produced.

    **One corpus-wide search, not one search per candidate.** Filtering retrieval down to a
    single document makes rank fusion degenerate — every hit is rank 1, so every candidate
    scores identically. The same bug distorted the ranking service and the first ablation.

    Screening is a filter for eligibility, not a second opinion on quality; ranking does
    quality. Where the signal is genuinely absent the screener passes rather than
    eliminates: a false negative removes someone from consideration entirely, which is the
    more costly error in a hiring tool.
    """

    # Evidence below this is treated as absent regardless of what the pool looks like.
    EVIDENCE_FLOOR = 0.15
    # A candidate scoring below this fraction of the pool's best is treated as lacking a
    # capability the rest of the pool demonstrably has.
    RELATIVE_FLOOR = 0.45
    # Terms too common in tech resumes to carry a must-have on their own.
    GENERIC_TERMS = {"python", "sql", "api", "apis", "data", "software", "cloud", "code", "ml", "ai"}

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm)

    def screen(
        self,
        *,
        role_requirements: Sequence[Requirement],
        candidates: Sequence[CandidateRecord],
        vectorstore: Any | None = None,
    ) -> list[ScreenResult]:
        must_haves = [req for req in role_requirements if req.must_have]

        if not must_haves or not candidates:
            return [
                ScreenResult(
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    screen_pass=True,
                    fail_reasons=[],
                    notes="No must-have requirements were extracted; every candidate proceeds to ranking.",
                )
                for candidate in candidates
            ]

        pool_scores: dict[str, dict[str, float]] = {}
        pool_text: dict[str, dict[str, str]] = {}
        for requirement in must_haves:
            scores, texts = self._pool_evidence(requirement, vectorstore)
            key = requirement.id or requirement.text
            pool_scores[key] = scores
            pool_text[key] = texts

        results: list[ScreenResult] = []
        for candidate in candidates:
            fail_reasons = [
                requirement.text
                for requirement in must_haves
                if not self._satisfies(requirement, candidate, pool_scores, pool_text)
            ]
            results.append(
                ScreenResult(
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    screen_pass=not fail_reasons,
                    fail_reasons=fail_reasons,
                    notes=(
                        f"Passed all {len(must_haves)} must-have check(s)."
                        if not fail_reasons
                        else f"Failed {len(fail_reasons)} of {len(must_haves)} must-have(s)."
                    ),
                )
            )
        return results

    # -- internals ---------------------------------------------------------

    def _pool_evidence(
        self, requirement: Requirement, vectorstore: Any | None
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Best score and best matching text per candidate, from one corpus-wide search."""
        if vectorstore is None:
            return {}, {}

        grouped_search = getattr(vectorstore, "search_grouped", None)
        if callable(grouped_search):
            grouped = grouped_search(
                query=requirement.text, per_entity_k=2, filters={"entity_type": "candidate"}
            )
        else:
            hits = vectorstore.search(
                query=requirement.text, k=200, filters={"entity_type": "candidate"}, fetch_k=200
            )
            grouped = {}
            for hit in hits:
                entity_id = hit.get("metadata", {}).get("entity_id")
                if entity_id is not None:
                    grouped.setdefault(entity_id, []).append(hit)

        scores = {
            entity_id: max((hit["score"] for hit in hits), default=0.0)
            for entity_id, hits in grouped.items()
        }
        texts = {
            entity_id: " ".join(hit.get("content", "") for hit in hits).lower()
            for entity_id, hits in grouped.items()
        }
        return scores, texts

    def _satisfies(
        self,
        requirement: Requirement,
        candidate: CandidateRecord,
        pool_scores: dict[str, dict[str, float]],
        pool_text: dict[str, dict[str, str]],
    ) -> bool:
        key = requirement.id or requirement.text
        distinctive = self._key_terms(requirement.text) - self.GENERIC_TERMS

        # Structured signal first: a distinctive skill extracted at ingest is precise and
        # free. Generic tokens are excluded — "python" appearing in both the requirement
        # and the skill list says nothing about "3+ years of professional Python".
        if distinctive and any(skill.lower() in distinctive for skill in candidate.skills):
            return True

        scores = pool_scores.get(key, {})
        if not scores:
            # No retrieval signal at all: pass rather than eliminate on no information.
            return True

        own = scores.get(candidate.id, 0.0)
        best_in_pool = max(scores.values(), default=0.0)

        weak_absolutely = own < self.EVIDENCE_FLOOR
        weak_relatively = best_in_pool > 0 and (own / best_in_pool) < self.RELATIVE_FLOOR
        if weak_absolutely or weak_relatively:
            return self._llm_adjudicates(requirement, pool_text.get(key, {}).get(candidate.id, ""))
        return True

    def _llm_adjudicates(self, requirement: Requirement, evidence: str) -> bool:
        """Last resort for ambiguous cases. With no LLM configured the candidate fails —
        retrieval has already said the evidence is not there."""
        if not self._llm or not evidence:
            return False
        reply = self._call_llm(
            f"Requirement: {requirement.text}\n"
            f"Resume evidence: {evidence[:800]}\n"
            "Does the evidence satisfy the requirement? Reply only 'yes' or 'no'.",
            max_tokens=8,
        )
        return reply.strip().lower().startswith("yes")

    @staticmethod
    def _key_terms(requirement: str) -> set[str]:
        """Content terms, minus stopwords and numerals.

        Numerals are dropped because "3+" in "3+ years of professional Python" will never
        appear literally in a resume that says "five years" — matching on it is noise.
        """
        tokens = re.findall(r"[a-zA-Z0-9+#./-]+", requirement.lower())
        return {
            token
            for token in tokens
            if token not in _STOP and len(token) > 1 and not token[0].isdigit()
        }
