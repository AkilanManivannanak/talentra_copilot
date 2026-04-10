from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Sequence

from app.models.schemas import CopilotAnswerResponse, Evidence, RequirementAssessment, RoleRecord


class SummaryService:
    def __init__(self, *, chat_model: str, openai_api_key: str) -> None:
        self._mode = "local"

    def candidate_summary(self, *, role: RoleRecord, candidate_name: str, assessments: Sequence[RequirementAssessment]) -> str:
        if not assessments:
            return f"{candidate_name} could not be evaluated because no role requirements were available."

        strongest = sorted(assessments, key=lambda item: item.score, reverse=True)[:2]
        covered = [a for a in assessments if a.covered]
        missing = [a.requirement.text for a in assessments if not a.covered][:2]

        strength_bits = [f"{assessment.requirement.text} ({assessment.score:.2f})" for assessment in strongest if assessment.score > 0]
        if covered:
            opener = f"{candidate_name} matches {len(covered)} of {len(assessments)} tracked requirements for {role.title}."
        else:
            opener = f"{candidate_name} does not show strong evidence against the core requirements for {role.title}."
        evidence_sentence = (
            f"Strongest signals: {', '.join(strength_bits)}."
            if strength_bits
            else "Strongest signals were weak or ambiguous in the ingested resume evidence."
        )
        gap_sentence = f"Main gaps: {'; '.join(missing)}." if missing else "No major gaps were detected from the retrieved evidence."
        return " ".join([opener, evidence_sentence, gap_sentence])

    def answer_question(self, *, question: str, citations: Sequence[Evidence], role_name: str) -> CopilotAnswerResponse:
        reasoning_trace = [
            f"Retrieved {len(citations)} evidence chunk(s).",
            f"Used local lexical retrieval and rule-based synthesis for role: {role_name}.",
        ]
        if not citations:
            return CopilotAnswerResponse(
                answer="I could not find supporting evidence for that question in the currently indexed role and candidate documents.",
                citations=[],
                reasoning_trace=reasoning_trace,
            )

        by_entity: dict[str, list[Evidence]] = defaultdict(list)
        for citation in citations:
            by_entity[citation.entity_name].append(citation)

        ranked_entities = sorted(by_entity.items(), key=lambda item: mean(c.score for c in item[1]), reverse=True)
        summaries = []
        for entity_name, entity_evidence in ranked_entities[:3]:
            top = sorted(entity_evidence, key=lambda item: item.score, reverse=True)[:2]
            avg_score = mean(item.score for item in top)
            snippets = " ".join(item.snippet for item in top)
            summaries.append(f"{entity_name} ({avg_score:.2f}): {snippets}")

        answer = f"Grounded findings for '{question}': " + " ".join(summaries)
        return CopilotAnswerResponse(answer=answer, citations=list(citations), reasoning_trace=reasoning_trace)
