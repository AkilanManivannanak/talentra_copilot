from __future__ import annotations

import re
from statistics import mean
from typing import Iterable

from app.models.schemas import CandidateEvaluation, CopilotAnswerResponse, Evidence
from app.services.metadata_store import MetadataStore
from app.services.ranking import RankingService
from app.services.summary import SummaryService
from app.services.vectorstore import VectorStoreService

_GENERIC_QUESTION_WORDS = {
    "who", "which", "candidate", "candidates", "role", "job", "position", "requirement", "requirements",
    "the", "a", "an", "is", "are", "for", "this", "that", "based", "results", "evaluation", "show",
    "shows", "showing", "strong", "stronger", "strongest", "best", "top", "match", "matches", "matched",
    "more", "most", "compare", "comparison", "give", "me", "why", "rank", "ranked", "above", "below",
    "evidence", "does", "has", "have", "what", "explain", "using", "current", "against", "fit", "fits",
    "candidate", "job", "description", "jd", "result", "results", "tell", "about", "there", "their",
}
_GENERIC_NAME_TOKENS = {"resume", "cv", "main", "jul2025", "candidate"}
_SKILL_HINTS = {"python", "machine learning", "ml", "ai", "rag", "retrieval", "fastapi", "docker", "sql", "api", "backend", "nlp", "llm", "cloud", "aws", "gcp"}


class CopilotService:
    def __init__(
        self,
        *,
        metadata: MetadataStore,
        vectorstore: VectorStoreService,
        summary_service: SummaryService,
        ranking_service: RankingService,
    ) -> None:
        self._metadata = metadata
        self._vectorstore = vectorstore
        self._summary = summary_service
        self._ranking = ranking_service

    def answer(self, *, question: str, role_id: str, candidate_ids: list[str] | None = None, top_k: int = 8) -> CopilotAnswerResponse:
        role = self._metadata.get_role(role_id)
        evaluation = self._ranking.evaluate_role(role_id=role_id, candidate_ids=candidate_ids or [], top_k_per_requirement=2)
        candidates = evaluation.candidates
        if not candidates:
            return CopilotAnswerResponse(
                answer="No candidates are available for this role yet.",
                citations=[],
                reasoning_trace=["No candidates were available for evaluation."],
            )

        question_clean = question.strip()
        if not question_clean:
            return CopilotAnswerResponse(answer="Ask a question first.", citations=[], reasoning_trace=[])

        mentioned = self._mentioned_candidates(question_clean, candidates)
        if self._is_evaluation_question(question_clean):
            return self._answer_from_evaluations(question=question_clean, role_name=role.title, candidates=candidates, mentioned=mentioned)

        focus_query = self._extract_focus_query(question_clean, candidates, role.title)
        if focus_query:
            targeted = self._answer_from_targeted_query(
                question=question_clean,
                role_name=role.title,
                candidates=candidates,
                focus_query=focus_query,
                top_k=top_k,
                mentioned=mentioned,
            )
            if targeted:
                return targeted

        citations = self._collect_generic_citations(question_clean, role_id, candidate_ids or [], top_k)
        return self._summary.answer_question(question=question_clean, citations=citations, role_name=role.title)

    def _is_evaluation_question(self, question: str) -> bool:
        q = question.lower()
        patterns = [
            "based on the evaluation",
            "strongest candidate",
            "best candidate",
            "top candidate",
            "matches the most",
            "most job requirements",
            "compare",
            "rank",
            "ranked",
            "above",
            "below",
            "fit the role",
            "fits the role",
            "fits best",
            "why is",
            "why does",
            "gaps",
            "missing requirements",
            "job requirements",
            "against the role",
        ]
        return any(pattern in q for pattern in patterns)

    def _mentioned_candidates(self, question: str, candidates: Iterable[CandidateEvaluation]) -> list[CandidateEvaluation]:
        q = re.sub(r"[^a-z0-9 ]+", " ", question.lower())
        found: list[CandidateEvaluation] = []
        for candidate in candidates:
            tokens = [
                token for token in re.sub(r"[^a-z0-9 ]+", " ", candidate.candidate_name.lower()).split()
                if token and token not in _GENERIC_NAME_TOKENS and len(token) > 2
            ]
            if any(token in q for token in tokens):
                found.append(candidate)
        return found

    def _extract_focus_query(self, question: str, candidates: Iterable[CandidateEvaluation], role_name: str) -> str:
        q = re.sub(r"[^a-z0-9+#. ]+", " ", question.lower())
        name_tokens = {
            token
            for candidate in candidates
            for token in re.sub(r"[^a-z0-9 ]+", " ", candidate.candidate_name.lower()).split()
            if token not in _GENERIC_NAME_TOKENS
        }
        explicit_skill_phrases = [phrase for phrase in _SKILL_HINTS if phrase in q]
        tokens = [
            token
            for token in q.split()
            if token not in _GENERIC_QUESTION_WORDS and token not in name_tokens and len(token) > 1
        ]
        compact = " ".join(tokens[:8]).strip()
        if explicit_skill_phrases:
            return " ".join(explicit_skill_phrases)
        return compact if compact and compact != role_name.lower() else ""

    def _answer_from_evaluations(
        self,
        *,
        question: str,
        role_name: str,
        candidates: list[CandidateEvaluation],
        mentioned: list[CandidateEvaluation],
    ) -> CopilotAnswerResponse:
        reasoning_trace = [
            f"Loaded evaluation results for {len(candidates)} candidate(s).",
            "Detected a ranking/comparison question and used evaluation results as the primary source of truth.",
        ]
        ranked = sorted(candidates, key=lambda item: (item.overall_score, item.matched_requirements), reverse=True)
        q = question.lower()

        if len(mentioned) >= 2 and any(token in q for token in ("above", "below", "compare", "why")):
            left, right = mentioned[0], mentioned[1]
            winner, loser = (left, right) if left.overall_score >= right.overall_score else (right, left)
            intro = (
                f"{winner.candidate_name} is ranked above {loser.candidate_name}" if winner is left else
                f"{left.candidate_name} is not actually ranked above {right.candidate_name}; {right.candidate_name} currently ranks higher"
            )
            answer = (
                f"{intro} for {role_name}. {winner.candidate_name} scored {winner.overall_score:.2f} with {winner.matched_requirements} matched requirements, "
                f"while {loser.candidate_name} scored {loser.overall_score:.2f} with {loser.matched_requirements} matched requirements. "
                f"{self._strength_sentence(winner)} {self._gap_sentence(loser)}"
            )
            citations = self._comparison_citations(winner, loser)
            reasoning_trace.append("Compared the named candidates directly from their requirement-level evaluations.")
            return CopilotAnswerResponse(answer=answer, citations=citations, reasoning_trace=reasoning_trace)

        if mentioned and ("gap" in q or "missing" in q):
            candidate = mentioned[0]
            gaps = [a.requirement.text for a in candidate.assessments if not a.covered][:3]
            answer = (
                f"For {role_name}, {candidate.candidate_name} is missing or weak on: {', '.join(gaps)}."
                if gaps else
                f"{candidate.candidate_name} does not show major gaps against the currently extracted requirements for {role_name}."
            )
            reasoning_trace.append("Answered a candidate-specific gap question from evaluation assessments.")
            return CopilotAnswerResponse(answer=answer, citations=self._top_evaluation_citations(candidate, covered=False), reasoning_trace=reasoning_trace)

        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        answer = (
            f"Based on the evaluation results, {winner.candidate_name} is the strongest candidate for {role_name}. "
            f"They scored {winner.overall_score:.2f} and matched {winner.matched_requirements} requirement(s)."
        )
        if runner_up:
            answer += (
                f" The next closest candidate is {runner_up.candidate_name} at {runner_up.overall_score:.2f}. "
                f"{self._strength_sentence(winner)}"
            )
        else:
            answer += f" {self._strength_sentence(winner)}"
        reasoning_trace.append("Returned the top-ranked candidate from the evaluation pipeline.")
        return CopilotAnswerResponse(answer=answer, citations=self._top_evaluation_citations(winner), reasoning_trace=reasoning_trace)

    def _answer_from_targeted_query(
        self,
        *,
        question: str,
        role_name: str,
        candidates: list[CandidateEvaluation],
        focus_query: str,
        top_k: int,
        mentioned: list[CandidateEvaluation],
    ) -> CopilotAnswerResponse | None:
        reasoning_trace = [
            f"Built a focused evidence query: '{focus_query}'.",
            "Compared candidates on targeted evidence rather than overall ranking.",
        ]
        targets = mentioned or candidates
        candidate_hits: list[tuple[CandidateEvaluation, list[Evidence], float]] = []
        for candidate in targets:
            hits = self._vectorstore.search(
                query=focus_query,
                k=min(3, top_k),
                filters={"entity_type": "candidate", "entity_id": candidate.candidate_id},
                fetch_k=max(8, top_k),
            )
            evidence = [
                Evidence(
                    document_id=item["metadata"]["document_id"],
                    filename=item["metadata"]["filename"],
                    entity_id=item["metadata"]["entity_id"],
                    entity_name=item["metadata"]["entity_name"],
                    snippet=item.get("snippet") or item["content"][:300],
                    score=item["score"],
                )
                for item in hits[:3]
            ]
            if evidence:
                candidate_hits.append((candidate, self._dedupe_citations(evidence), mean(item.score for item in evidence)))

        if not candidate_hits:
            return None

        candidate_hits.sort(key=lambda item: (item[2], item[0].overall_score), reverse=True)
        winner, citations, avg_score = candidate_hits[0]
        answer = (
            f"For the focused question '{question}', {winner.candidate_name} shows the strongest direct evidence. "
            f"Their targeted evidence score is {avg_score:.2f}."
        )
        if len(candidate_hits) > 1:
            runner_up, _, runner_score = candidate_hits[1]
            answer += f" The next closest candidate is {runner_up.candidate_name} at {runner_score:.2f}."
        answer += f" This is evidence for the specific topic '{focus_query}', not the overall role ranking."
        return CopilotAnswerResponse(answer=answer, citations=citations, reasoning_trace=reasoning_trace)

    def _collect_generic_citations(self, question: str, role_id: str, candidate_ids: list[str], top_k: int) -> list[Evidence]:
        citations: list[Evidence] = []
        if candidate_ids:
            for candidate_id in candidate_ids:
                hits = self._vectorstore.search(query=question, k=top_k, filters={"entity_type": "candidate", "entity_id": candidate_id})
                citations.extend(self._evidence_from_hits(hits))
        else:
            hits = self._vectorstore.search(query=question, k=top_k, filters={"entity_type": "candidate"})
            citations.extend(self._evidence_from_hits(hits))
        return self._dedupe_citations(citations)[:top_k]

    def _evidence_from_hits(self, hits: list[dict]) -> list[Evidence]:
        return [
            Evidence(
                document_id=item["metadata"]["document_id"],
                filename=item["metadata"]["filename"],
                entity_id=item["metadata"]["entity_id"],
                entity_name=item["metadata"]["entity_name"],
                snippet=item.get("snippet") or item["content"][:320],
                score=item["score"],
            )
            for item in hits
        ]

    def _top_evaluation_citations(self, candidate: CandidateEvaluation, covered: bool | None = None) -> list[Evidence]:
        assessments = candidate.assessments
        if covered is True:
            assessments = [item for item in assessments if item.covered]
        elif covered is False:
            assessments = [item for item in assessments if not item.covered]
        evidence = [citation for assessment in assessments for citation in assessment.evidence]
        evidence.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_citations(evidence)[:5]

    def _comparison_citations(self, winner: CandidateEvaluation, loser: CandidateEvaluation) -> list[Evidence]:
        citations = self._top_evaluation_citations(winner)[:3] + self._top_evaluation_citations(loser)[:2]
        return self._dedupe_citations(citations)[:5]

    def _dedupe_citations(self, citations: list[Evidence]) -> list[Evidence]:
        seen: set[tuple[str, str]] = set()
        deduped: list[Evidence] = []
        for citation in citations:
            key = (citation.entity_id, citation.snippet.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
        return deduped

    def _strength_sentence(self, candidate: CandidateEvaluation) -> str:
        strengths = [a for a in sorted(candidate.assessments, key=lambda item: item.score, reverse=True) if a.score > 0][:2]
        if not strengths:
            return f"The retrieved evidence for {candidate.candidate_name} is generally weak across the tracked requirements."
        parts = [f"{item.requirement.text} ({item.score:.2f})" for item in strengths]
        return f"Their strongest requirement coverage is {', '.join(parts)}."

    def _gap_sentence(self, candidate: CandidateEvaluation) -> str:
        gaps = [a.requirement.text for a in candidate.assessments if not a.covered][:2]
        if not gaps:
            return f"{candidate.candidate_name} does not show major uncovered requirements in the current evaluation."
        return f"Main weaker areas for {candidate.candidate_name}: {', '.join(gaps)}."
