from __future__ import annotations

from collections import Counter

from app.models.schemas import ATSDashboardResponse, CandidatePipelineView, RecruiterNote, StageCount
from app.services.metadata_store import MetadataStore
from app.services.ranking import RankingService

_STAGE_ORDER = ["Applied", "Screening", "Shortlisted", "Interview", "Final", "Rejected"]


class ATSService:
    def __init__(self, *, metadata: MetadataStore, ranking: RankingService) -> None:
        self._metadata = metadata
        self._ranking = ranking

    def dashboard(self, *, role_id: str, candidate_ids: list[str] | None = None) -> ATSDashboardResponse:
        evaluation = self._ranking.evaluate_role(role_id=role_id, candidate_ids=candidate_ids or [], top_k_per_requirement=2)
        candidate_index = {candidate.id: candidate for candidate in self._metadata.list_candidates()}
        pipeline: list[CandidatePipelineView] = []

        for item in evaluation.candidates:
            candidate = candidate_index.get(item.candidate_id)
            if candidate is None:
                continue
            pipeline.append(
                CandidatePipelineView(
                    candidate_id=item.candidate_id,
                    candidate_name=item.candidate_name,
                    stage=candidate.stage,
                    shortlisted=candidate.shortlisted,
                    notes_count=self._metadata.count_notes(item.candidate_id),
                    latest_score=round(item.overall_score, 2),
                    matched_requirements=item.matched_requirements,
                    missing_requirements=item.missing_requirements,
                    summary=item.summary,
                )
            )

        pipeline.sort(
            key=lambda row: (
                not row.shortlisted,
                _STAGE_ORDER.index(row.stage) if row.stage in _STAGE_ORDER else 999,
                -(row.latest_score or 0.0),
                row.candidate_name.lower(),
            )
        )
        counts = Counter(item.stage for item in pipeline)
        stage_counts = [StageCount(stage=stage, count=counts.get(stage, 0)) for stage in _STAGE_ORDER]
        shortlist = [item for item in pipeline if item.shortlisted or item.stage in {"Shortlisted", "Interview", "Final"}]
        return ATSDashboardResponse(role=evaluation.role, stage_counts=stage_counts, candidates=pipeline, shortlist=shortlist)

    def update_stage(self, *, candidate_id: str, stage: str):
        return self._metadata.update_candidate_stage(candidate_id, stage)

    def update_shortlist(self, *, candidate_id: str, shortlisted: bool):
        return self._metadata.update_candidate_shortlist(candidate_id, shortlisted)

    def add_note(self, *, candidate_id: str, text: str) -> RecruiterNote:
        return self._metadata.add_note(candidate_id, text)

    def list_notes(self, *, candidate_id: str) -> list[RecruiterNote]:
        return self._metadata.list_notes(candidate_id)
