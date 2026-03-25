from __future__ import annotations

from statistics import mean

from app.core.config import Settings
from app.models.schemas import CandidateEvaluation, EvaluateRoleResponse, Evidence, RequirementAssessment
from app.services.metadata_store import MetadataStore
from app.services.summary import SummaryService
from app.services.vectorstore import VectorStoreService


class RankingService:
    def __init__(
        self,
        *,
        settings: Settings,
        metadata: MetadataStore,
        vectorstore: VectorStoreService,
        summary_service: SummaryService,
    ) -> None:
        self._settings = settings
        self._metadata = metadata
        self._vectorstore = vectorstore
        self._summary = summary_service
        self._cache: dict[tuple, EvaluateRoleResponse] = {}

    def evaluate_role(
        self,
        *,
        role_id: str,
        candidate_ids: list[str] | None = None,
        top_k_per_requirement: int = 2,
    ) -> EvaluateRoleResponse:
        cache_key = (
            role_id,
            tuple(sorted(candidate_ids or [])),
            top_k_per_requirement,
            getattr(self._metadata, "version", 0),
            getattr(self._vectorstore, "version", 0),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(deep=True)

        role = self._metadata.get_role(role_id)
        candidates = self._metadata.list_candidates()
        if candidate_ids:
            wanted = set(candidate_ids)
            candidates = [candidate for candidate in candidates if candidate.id in wanted]

        evaluations: list[CandidateEvaluation] = []
        for candidate in candidates:
            assessments: list[RequirementAssessment] = []
            for requirement in role.requirements:
                hits = self._vectorstore.search(
                    query=requirement.text,
                    k=top_k_per_requirement,
                    filters={"entity_type": "candidate", "entity_id": candidate.id},
                )[:top_k_per_requirement]
                evidence = [
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
                if evidence:
                    max_score = max(item.score for item in evidence)
                    avg_score = mean(item.score for item in evidence)
                    score = (0.75 * max_score) + (0.25 * avg_score)
                else:
                    score = 0.0
                assessments.append(
                    RequirementAssessment(
                        requirement=requirement,
                        score=max(0.0, min(score, 1.0)),
                        covered=score >= self._settings.match_threshold,
                        evidence=evidence,
                    )
                )

            weighted_scores = [a.score * a.requirement.weight for a in assessments]
            total_weight = sum(a.requirement.weight for a in assessments) or 1.0
            overall_score = sum(weighted_scores) / total_weight
            matched = sum(1 for a in assessments if a.covered)
            summary = self._summary.candidate_summary(role=role, candidate_name=candidate.name, assessments=assessments)
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    overall_score=max(0.0, min(overall_score, 1.0)),
                    matched_requirements=matched,
                    missing_requirements=max(0, len(assessments) - matched),
                    summary=summary,
                    assessments=assessments,
                )
            )

        evaluations.sort(
            key=lambda item: (item.overall_score, item.matched_requirements, -item.missing_requirements),
            reverse=True,
        )
        response = EvaluateRoleResponse(role=role, candidates=evaluations)
        if len(self._cache) >= self._settings.evaluation_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = response.model_copy(deep=True)
        return response
