"""
LangGraph orchestration for the Talentra hiring workflow.

    screen -> evaluate -> audit_bias -> route_question -> copilot_qa | evidence_search
                                                      -> ats_update  [interrupt]

Two things make this a real graph rather than a diagram:

1. **The nodes call the same services the REST API calls.** They are built by
   `build_hiring_graph(services)` and close over the live `ServiceContainer`, so the
   agentic path and the plain path cannot drift apart or produce different rankings.
   (The previous version constructed `RankerAgent()` with no vectorstore, which scored
   every candidate 0.)

2. **The human-in-the-loop interrupt actually blocks.** The graph is compiled with a
   checkpointer and `interrupt_before=["ats_update"]`, so `invoke` returns with the ATS
   write *not applied* and the run parked. It resumes only when a recruiter approves,
   via `resume_after_approval`. Approval is not a log line; nothing is written without it.

If `langgraph` is not installed the module falls back to `_SequentialGraph`, which runs
the same node functions in the same order and honours the same interrupt contract. The
fallback is reported honestly at `/ops/build` as `langgraph_available: false` — it is a
degradation, not an equivalent.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

Route = Literal["copilot_qa", "evidence_search", "done"]

_COMPARISON_KEYWORDS = (
    "compare", "versus", " vs ", "rank", "ranked", "strongest", "best candidate",
    "why is", "why does", "above", "below", "gaps", "missing",
)


class HiringState(TypedDict, total=False):
    # inputs
    role_id: str
    role_title: str
    candidate_ids: list[str]
    question: str | None
    top_k: int
    run_bias_audit: bool
    ats_action: dict[str, Any] | None

    # node outputs
    screening: list[dict]
    evaluation: dict | None
    bias_audit: dict | None
    route: Route
    answer: dict | None
    ats_committed: bool

    # bookkeeping
    nodes_executed: list[str]
    errors: list[str]


def _mark(state: HiringState, node: str) -> list[str]:
    return list(state.get("nodes_executed", [])) + [node]


# ---------------------------------------------------------------------------
# Node factories — each closes over the live ServiceContainer
# ---------------------------------------------------------------------------

def make_screen_node(services):
    from app.agents.screener import ScreenerAgent

    def screen_node(state: HiringState) -> HiringState:
        errors = list(state.get("errors", []))
        try:
            role = services.metadata.get_role(state["role_id"])
            agent = ScreenerAgent()
            results = agent.screen(
                role_requirements=role.requirements,
                candidates=[services.metadata.get_candidate(cid) for cid in state.get("candidate_ids", [])],
                vectorstore=services.retriever,
            )
            screening = [item.model_dump() for item in results]
        except Exception as exc:
            logger.exception("screen_node failed")
            errors.append(f"screen: {type(exc).__name__}: {exc}")
            screening = []
        return {**state, "screening": screening, "errors": errors, "nodes_executed": _mark(state, "screen")}

    return screen_node


def make_evaluate_node(services):
    def evaluate_node(state: HiringState) -> HiringState:
        errors = list(state.get("errors", []))
        evaluation = None
        try:
            # Candidates that failed a hard must-have are excluded from ranking rather
            # than ranked low — that is the entire point of having a screening stage.
            passed = [item["candidate_id"] for item in state.get("screening", []) if item.get("screen_pass")]
            candidate_ids = passed or state.get("candidate_ids", [])
            response = services.ranking.evaluate_role(
                role_id=state["role_id"],
                candidate_ids=candidate_ids,
                top_k_per_requirement=state.get("top_k", 3),
            )
            evaluation = response.model_dump(mode="json")
        except Exception as exc:
            logger.exception("evaluate_node failed")
            errors.append(f"evaluate: {type(exc).__name__}: {exc}")
        return {**state, "evaluation": evaluation, "errors": errors, "nodes_executed": _mark(state, "evaluate")}

    return evaluate_node


def make_bias_audit_node(services):
    from app.models.schemas import EvaluateRoleResponse

    def bias_audit_node(state: HiringState) -> HiringState:
        if not state.get("run_bias_audit", True) or not state.get("evaluation"):
            return {**state, "nodes_executed": _mark(state, "audit_bias:skipped")}
        errors = list(state.get("errors", []))
        report = None
        try:
            evaluation = EvaluateRoleResponse.model_validate(state["evaluation"])
            report = services.bias_auditor.audit(
                requirements=evaluation.role.requirements,
                evaluations=evaluation.candidates,
            ).model_dump(mode="json")
        except Exception as exc:
            logger.exception("bias_audit_node failed")
            errors.append(f"bias_audit: {type(exc).__name__}: {exc}")
        return {**state, "bias_audit": report, "errors": errors, "nodes_executed": _mark(state, "audit_bias")}

    return bias_audit_node


def route_question(state: HiringState) -> Route:
    """Conditional edge.

    This mirrors the v4->v5 postmortem fix: comparison and ranking questions must be
    answered from the evaluation results, never from raw retrieval, or the Copilot can
    contradict the ranker it is supposed to be explaining.
    """
    question = (state.get("question") or "").strip().lower()
    if not question:
        return "done"
    if any(keyword in question for keyword in _COMPARISON_KEYWORDS):
        return "copilot_qa"
    return "evidence_search"


def make_copilot_node(services):
    def copilot_qa_node(state: HiringState) -> HiringState:
        errors = list(state.get("errors", []))
        answer = None
        try:
            response = services.copilot.answer(
                question=state.get("question") or "",
                role_id=state["role_id"],
                candidate_ids=state.get("candidate_ids", []),
                top_k=max(4, state.get("top_k", 3) * 2),
            )
            answer = response.model_dump(mode="json")
        except Exception as exc:
            logger.exception("copilot_qa_node failed")
            errors.append(f"copilot_qa: {type(exc).__name__}: {exc}")
        return {
            **state, "answer": answer, "route": "copilot_qa",
            "errors": errors, "nodes_executed": _mark(state, "copilot_qa"),
        }

    return copilot_qa_node


def make_evidence_node(services):
    from app.models.schemas import Evidence

    def evidence_search_node(state: HiringState) -> HiringState:
        errors = list(state.get("errors", []))
        answer = None
        try:
            question = state.get("question") or ""
            citations: list[Evidence] = []
            for candidate_id in state.get("candidate_ids", []):
                hits = services.retriever.search(
                    query=question,
                    k=3,
                    filters={"entity_type": "candidate", "entity_id": candidate_id},
                    fetch_k=20,
                )
                for hit in hits:
                    citations.append(
                        Evidence(
                            document_id=hit["metadata"]["document_id"],
                            filename=hit["metadata"]["filename"],
                            entity_id=hit["metadata"]["entity_id"],
                            entity_name=hit["metadata"]["entity_name"],
                            snippet=hit.get("snippet") or hit["content"][:320],
                            score=hit["score"],
                            chunk_id=hit["metadata"].get("chunk_id", ""),
                        )
                    )
            citations.sort(key=lambda item: item.score, reverse=True)
            citations = citations[: max(4, state.get("top_k", 3) * 2)]
            response = services.summary.answer_question(
                question=question, citations=citations, role_name=state.get("role_title", "")
            )
            if services.faithfulness is not None:
                response.faithfulness = services.faithfulness.score(
                    answer=response.answer, citations=response.citations
                )
            answer = response.model_dump(mode="json")
        except Exception as exc:
            logger.exception("evidence_search_node failed")
            errors.append(f"evidence_search: {type(exc).__name__}: {exc}")
        return {
            **state, "answer": answer, "route": "evidence_search",
            "errors": errors, "nodes_executed": _mark(state, "evidence_search"),
        }

    return evidence_search_node


def make_ats_node(services):
    def ats_update_node(state: HiringState) -> HiringState:
        """Runs only after a recruiter approves; the graph is interrupted before this node."""
        action = state.get("ats_action")
        errors = list(state.get("errors", []))
        if not action:
            return {**state, "ats_committed": False, "nodes_executed": _mark(state, "ats_update:noop")}
        try:
            candidate_id = action["candidate_id"]
            if "stage" in action:
                services.ats.update_stage(candidate_id=candidate_id, stage=action["stage"])
            if "shortlisted" in action:
                services.ats.update_shortlist(candidate_id=candidate_id, shortlisted=bool(action["shortlisted"]))
            if action.get("note"):
                services.ats.add_note(candidate_id=candidate_id, text=action["note"])
            committed = True
        except Exception as exc:
            logger.exception("ats_update_node failed")
            errors.append(f"ats_update: {type(exc).__name__}: {exc}")
            committed = False
        return {**state, "ats_committed": committed, "errors": errors, "nodes_executed": _mark(state, "ats_update")}

    return ats_update_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def langgraph_available() -> bool:
    try:
        import langgraph  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def build_hiring_graph(services, *, interrupt_before_ats: bool = True):
    nodes = {
        "screen": make_screen_node(services),
        "evaluate": make_evaluate_node(services),
        "audit_bias": make_bias_audit_node(services),
        "copilot_qa": make_copilot_node(services),
        "evidence_search": make_evidence_node(services),
        "ats_update": make_ats_node(services),
    }

    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
        from langgraph.graph import END, StateGraph  # type: ignore
    except ImportError:
        logger.warning("langgraph not installed; using the sequential fallback executor.")
        return _SequentialGraph(nodes, interrupt_before_ats=interrupt_before_ats)

    graph = StateGraph(HiringState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("screen")
    graph.add_edge("screen", "evaluate")
    graph.add_edge("evaluate", "audit_bias")
    graph.add_conditional_edges(
        "audit_bias",
        route_question,
        {"copilot_qa": "copilot_qa", "evidence_search": "evidence_search", "done": "ats_update"},
    )
    graph.add_edge("copilot_qa", "ats_update")
    graph.add_edge("evidence_search", "ats_update")
    graph.add_edge("ats_update", END)

    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["ats_update"] if interrupt_before_ats else [],
    )


class _SequentialGraph:
    """Fallback executor used only when langgraph is absent.

    It honours the same interrupt contract — stop before `ats_update`, keep the state,
    resume on approval — so the API behaves identically. It does not provide langgraph's
    checkpoint persistence, streaming, or retry semantics, and `/ops/build` says so.
    """

    def __init__(self, nodes: dict, *, interrupt_before_ats: bool = True) -> None:
        self._nodes = nodes
        self._interrupt = interrupt_before_ats
        self._parked: dict[str, HiringState] = {}

    def invoke(self, state: HiringState | None, config: dict | None = None) -> HiringState:
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        if state is None:  # resume
            state = self._parked.pop(thread_id, {})
            return self._nodes["ats_update"](state)

        state = self._nodes["screen"](state)
        state = self._nodes["evaluate"](state)
        state = self._nodes["audit_bias"](state)
        route = route_question(state)
        if route in ("copilot_qa", "evidence_search"):
            state = self._nodes[route](state)
        else:
            state = {**state, "route": "done"}

        if self._interrupt:
            self._parked[thread_id] = state
            return state
        return self._nodes["ats_update"](state)

    def get_state(self, config: dict) -> Any:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        parked = thread_id in self._parked

        class _Snapshot:
            next = ("ats_update",) if parked else ()
            values = self._parked.get(thread_id, {})

        return _Snapshot()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
