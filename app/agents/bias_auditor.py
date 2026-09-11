"""BiasAuditorAgent: thin agent wrapper over the counterfactual bias audit.

The previous implementation inferred gender from a hardcoded list of first names and
flagged group-mean gaps computed over as few as one candidate per group. That has been
removed. Talentra performs no demographic inference of any kind.

The audit logic now lives in `app.services.bias_audit.CounterfactualBiasAuditor`, which
measures how much each candidate's score depends on identity signals (school, employer,
name) by re-scoring the resume with those signals redacted. This class exists so the
graph has a consistent agent-shaped seam; it holds no scoring logic of its own.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.models.schemas import BiasAuditReport, CandidateEvaluation, Requirement
from app.services.bias_audit import CounterfactualBiasAuditor

from .base import BaseAgent


class BiasAuditorAgent(BaseAgent):
    def __init__(self, auditor: CounterfactualBiasAuditor, llm: Any | None = None) -> None:
        super().__init__(llm)
        self._auditor = auditor

    def audit(
        self,
        *,
        requirements: Sequence[Requirement],
        evaluations: Sequence[CandidateEvaluation],
    ) -> BiasAuditReport:
        return self._auditor.audit(requirements=requirements, evaluations=evaluations)
