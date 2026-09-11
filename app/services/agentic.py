"""Runner that owns the compiled hiring graph and the lifecycle of interrupted runs."""
from __future__ import annotations

import logging
import threading
from typing import Any

from app.graph.hiring_graph import build_hiring_graph, langgraph_available, new_run_id
from app.models.schemas import (
    AgenticEvaluateRequest,
    AgenticEvaluateResponse,
    AgenticResumeResponse,
    BiasAuditReport,
    CopilotAnswerResponse,
    EvaluateRoleResponse,
    ScreenResult,
)

logger = logging.getLogger(__name__)


class AgenticRunner:
    def __init__(self, services, *, interrupt_before_ats: bool = True) -> None:
        self._services = services
        self._interrupt_before_ats = interrupt_before_ats
        self._graph = build_hiring_graph(services, interrupt_before_ats=interrupt_before_ats)
        self._lock = threading.Lock()
        # run_id -> the action awaiting recruiter approval
        self._pending: dict[str, dict[str, Any]] = {}

    @property
    def langgraph_available(self) -> bool:
        return langgraph_available()

    def run(self, *, role_id: str, request: AgenticEvaluateRequest) -> AgenticEvaluateResponse:
        role = self._services.metadata.get_role(role_id)
        candidate_ids = request.candidate_ids or [c.id for c in self._services.metadata.list_candidates()]
        run_id = new_run_id()
        config = {"configurable": {"thread_id": run_id}}

        initial = {
            "role_id": role_id,
            "role_title": role.title,
            "candidate_ids": candidate_ids,
            "question": request.question,
            "top_k": request.top_k_per_requirement,
            "run_bias_audit": request.run_bias_audit,
            "ats_action": request.ats_action,
            "nodes_executed": [],
            "errors": [],
        }

        final = self._graph.invoke(initial, config)
        interrupted = self._is_interrupted(config)

        if interrupted and request.ats_action:
            with self._lock:
                self._pending[run_id] = request.ats_action

        return AgenticEvaluateResponse(
            role_id=role_id,
            role_title=role.title,
            run_id=run_id,
            route_taken=final.get("route", "done"),
            nodes_executed=final.get("nodes_executed", []),
            screening=[ScreenResult.model_validate(item) for item in final.get("screening", [])],
            evaluation=EvaluateRoleResponse.model_validate(final["evaluation"]) if final.get("evaluation") else None,
            answer=CopilotAnswerResponse.model_validate(final["answer"]) if final.get("answer") else None,
            bias_audit=BiasAuditReport.model_validate(final["bias_audit"]) if final.get("bias_audit") else None,
            interrupted_before="ats_update" if interrupted else None,
            pending_ats_action=request.ats_action if interrupted else None,
            errors=final.get("errors", []),
        )

    def resume(self, *, run_id: str, approve: bool) -> AgenticResumeResponse:
        config = {"configurable": {"thread_id": run_id}}
        with self._lock:
            action = self._pending.pop(run_id, None)

        if not approve:
            return AgenticResumeResponse(
                run_id=run_id,
                approved=False,
                ats_committed=False,
                applied_action=None,
                message="Recruiter declined. The run was discarded and no ATS write was made.",
            )
        if not self._is_interrupted(config):
            return AgenticResumeResponse(
                run_id=run_id,
                approved=True,
                ats_committed=False,
                applied_action=None,
                message="No run is parked under this run_id. It may have already been resumed or expired.",
            )

        # Passing None resumes from the checkpoint rather than starting a new run.
        final = self._graph.invoke(None, config)
        committed = bool(final.get("ats_committed"))
        return AgenticResumeResponse(
            run_id=run_id,
            approved=True,
            ats_committed=committed,
            applied_action=action,
            message=(
                "Recruiter approved; the ATS write was applied."
                if committed
                else "Recruiter approved, but no ATS action was attached to this run."
            ),
        )

    def _is_interrupted(self, config: dict) -> bool:
        try:
            snapshot = self._graph.get_state(config)
        except Exception:
            return False
        return bool(getattr(snapshot, "next", ()))
