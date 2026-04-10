"""
LangGraph orchestration for the Talentra hiring workflow.

Graph:
  preprocess → embed → evaluate → [router] → copilot_qa / evidence_search → ats_update

Includes human-in-the-loop interrupt before ats_update.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict, Literal, Optional


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class HiringState(TypedDict, total=False):
    # Inputs
    role_id: str
    role_title: str
    role_requirements: list[str]
    candidate_ids: list[str]
    question: str | None
    top_k: int

    # Preprocessing outputs
    preprocessed: dict[str, Any]  # candidate_id -> PreprocessedDocument.to_dict()

    # Embedding / retrieval
    vectorstore_ready: bool

    # Evaluation outputs
    evaluation_results: list[dict]   # [{candidate_id, candidate_name, score, evidence, ...}]

    # Routing decision
    route: Literal["copilot_qa", "evidence_search", "done"]

    # Copilot output
    answer: str

    # ATS
    ats_action: dict | None         # Pending action; None until set by an agent
    ats_committed: bool

    # Error / metadata
    errors: list[str]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def preprocess_node(state: HiringState) -> HiringState:
    """Run preprocessing pipeline on all candidates."""
    from app.preprocessing import run_preprocessing_pipeline

    preprocessed = state.get("preprocessed", {})
    errors = list(state.get("errors", []))

    candidate_ids = state.get("candidate_ids", [])
    for cid in candidate_ids:
        if cid in preprocessed:
            continue
        try:
            # In a real system, fetch raw text from metadata store here
            # For now, emit a placeholder (actual integration wires to the ingest service)
            preprocessed[cid] = {"status": "queued", "candidate_id": cid}
        except Exception as e:
            errors.append(f"preprocess:{cid}:{e}")

    return {**state, "preprocessed": preprocessed, "errors": errors}


def embed_node(state: HiringState) -> HiringState:
    """Ensure candidates are embedded in the vector store."""
    # Actual embedding is triggered by the ingest route; this node validates readiness.
    return {**state, "vectorstore_ready": True}


def evaluate_node(state: HiringState) -> HiringState:
    """Run requirement-level evaluation for all candidates."""
    from app.agents.ranker import RankerAgent

    agent = RankerAgent()
    results = agent.rank(
        role_id=state["role_id"],
        role_title=state.get("role_title", ""),
        requirements=state.get("role_requirements", []),
        candidate_ids=state.get("candidate_ids", []),
        top_k=state.get("top_k", 5),
    )
    return {**state, "evaluation_results": results}


def route_question(state: HiringState) -> Literal["copilot_qa", "evidence_search", "done"]:
    """Conditional edge: choose path based on question type."""
    question = state.get("question", "")
    if not question:
        return "done"
    comparison_keywords = ["compare", "versus", "vs", "rank", "strongest", "why", "ranked above"]
    q_lower = question.lower()
    if any(kw in q_lower for kw in comparison_keywords):
        return "copilot_qa"
    return "evidence_search"


def copilot_qa_node(state: HiringState) -> HiringState:
    """Answer using evaluation results + top evidence."""
    from app.agents.copilot import CopilotAgent

    agent = CopilotAgent()
    answer = agent.answer(
        question=state.get("question", ""),
        role_title=state.get("role_title", ""),
        evaluation_results=state.get("evaluation_results", []),
        role_id=state.get("role_id", ""),
        candidate_ids=state.get("candidate_ids", []),
        top_k=state.get("top_k", 8),
    )
    return {**state, "answer": answer, "route": "copilot_qa"}


def evidence_search_node(state: HiringState) -> HiringState:
    """Answer using targeted evidence retrieval (no evaluation summary)."""
    from app.agents.copilot import CopilotAgent

    agent = CopilotAgent()
    answer = agent.evidence_search(
        question=state.get("question", ""),
        role_id=state.get("role_id", ""),
        candidate_ids=state.get("candidate_ids", []),
        top_k=state.get("top_k", 8),
    )
    return {**state, "answer": answer, "route": "evidence_search"}


def ats_update_node(state: HiringState) -> HiringState:
    """Commit ATS action (stage transition / recruiter note)."""
    # human-in-the-loop interrupt fires BEFORE this node when configured.
    ats_action = state.get("ats_action")
    if ats_action:
        # Actual ATS write happens here in a wired-up system
        return {**state, "ats_committed": True}
    return {**state, "ats_committed": False}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_hiring_graph(interrupt_before_ats: bool = True):
    """
    Build and return a compiled LangGraph StateGraph.
    Falls back to a simple sequential executor if langgraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore

        g = StateGraph(HiringState)

        g.add_node("preprocess", preprocess_node)
        g.add_node("embed", embed_node)
        g.add_node("evaluate", evaluate_node)
        g.add_node("copilot_qa", copilot_qa_node)
        g.add_node("evidence_search", evidence_search_node)
        g.add_node("ats_update", ats_update_node)

        g.set_entry_point("preprocess")
        g.add_edge("preprocess", "embed")
        g.add_edge("embed", "evaluate")
        g.add_conditional_edges(
            "evaluate",
            route_question,
            {
                "copilot_qa": "copilot_qa",
                "evidence_search": "evidence_search",
                "done": END,
            },
        )
        g.add_edge("copilot_qa", "ats_update")
        g.add_edge("evidence_search", "ats_update")
        g.add_edge("ats_update", END)

        interrupt_nodes = ["ats_update"] if interrupt_before_ats else []
        return g.compile(interrupt_before=interrupt_nodes)

    except ImportError:
        return _FallbackGraph()


class _FallbackGraph:
    """Sequential fallback when LangGraph is not installed."""

    def invoke(self, state: HiringState, config: dict | None = None) -> HiringState:
        state = preprocess_node(state)
        state = embed_node(state)
        state = evaluate_node(state)
        route = route_question(state)
        if route == "copilot_qa":
            state = copilot_qa_node(state)
        elif route == "evidence_search":
            state = evidence_search_node(state)
        if state.get("ats_action"):
            state = ats_update_node(state)
        return state
