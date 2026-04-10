"""CopilotAgent: evaluation-aware Q&A and evidence search."""
from __future__ import annotations

import re
from typing import Any

from .base import BaseAgent


class CopilotAgent(BaseAgent):
    """
    Answers recruiter questions using:
      - Evaluation results (for comparison / ranking questions)
      - Direct evidence retrieval (for skill-specific questions)
    """

    def __init__(self, llm: Any | None = None, vectorstore: Any | None = None):
        super().__init__(llm)
        self._vectorstore = vectorstore

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        role_title: str,
        evaluation_results: list[dict],
        role_id: str = "",
        candidate_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        """
        Primary path: evaluation-aware answer.
        Uses evaluation summary + retrieved evidence.
        """
        eval_summary = self._format_eval_summary(evaluation_results)
        evidence_chunks = self._retrieve_evidence(question, candidate_ids or [], top_k)
        evidence_str = self._format_evidence(evidence_chunks)

        if self._llm:
            return self._llm_answer(question, role_title, eval_summary, evidence_str)

        return self._rule_based_answer(question, evaluation_results, evidence_chunks)

    def evidence_search(
        self,
        question: str,
        role_id: str = "",
        candidate_ids: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        """
        Secondary path: targeted evidence retrieval without evaluation context.
        Used for skill-specific or factual questions.
        """
        chunks = self._retrieve_evidence(question, candidate_ids or [], top_k)
        if not chunks:
            return (
                "I couldn't find specific evidence for that query in the uploaded resumes. "
                "Try rephrasing or check that the relevant resumes have been uploaded."
            )

        evidence_str = self._format_evidence(chunks)

        if self._llm:
            prompt = (
                f"Question: {question}\n\n"
                f"Evidence from resumes:\n{evidence_str}\n\n"
                "Answer the question using only the evidence above. "
                "Cite evidence by number when possible."
            )
            reply = self._call_llm(prompt, max_tokens=400)
            if reply.strip():
                return reply.strip()

        # Fallback: return formatted evidence with a header
        return (
            f"Here is the most relevant evidence I found for '{question}':\n\n"
            + evidence_str
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_eval_summary(self, evaluation_results: list[dict]) -> str:
        if not evaluation_results:
            return "No evaluation results available."
        lines = ["Ranked candidates (highest score first):"]
        for i, c in enumerate(evaluation_results, 1):
            name = c.get("candidate_name", c.get("candidate_id", "Unknown"))
            pct = c.get("pct_score", 0.0)
            lines.append(f"  {i}. {name} — {pct:.0%} match")
            for rs in c.get("requirement_scores", [])[:4]:
                label = rs.get("score_label", "")
                req = rs.get("requirement", "")[:60]
                lines.append(f"      • {req}: {label}")
        return "\n".join(lines)

    def _retrieve_evidence(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int,
    ) -> list[dict]:
        if self._vectorstore is None:
            return []
        chunks = []
        per_candidate = max(1, top_k // max(len(candidate_ids), 1))
        for cid in candidate_ids:
            try:
                results = self._vectorstore.search(query=query, doc_id=cid, top_k=per_candidate)
                chunks.extend(results)
            except Exception:
                pass
        # Sort by score descending
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return chunks[:top_k]

    def _llm_answer(
        self,
        question: str,
        role_title: str,
        eval_summary: str,
        evidence: str,
    ) -> str:
        from app.langchain_layer.prompts import COPILOT_PROMPT, format_prompt
        prompt = format_prompt(
            COPILOT_PROMPT,
            role_title=role_title,
            evaluation_summary=eval_summary,
            evidence=evidence,
            question=question,
        )
        reply = self._call_llm(prompt, max_tokens=600)
        return reply.strip() if reply.strip() else self._rule_based_answer(question, [], [])

    def _rule_based_answer(
        self,
        question: str,
        evaluation_results: list[dict],
        evidence_chunks: list[dict],
    ) -> str:
        """Deterministic fallback when no LLM is available."""
        q = question.lower()
        if not evaluation_results:
            return "No evaluation results are available. Please run an evaluation first."

        top = evaluation_results[0]
        top_name = top.get("candidate_name", "the top candidate")

        # "Who is the strongest / best candidate?"
        if any(w in q for w in ["strongest", "best", "top", "who should", "recommend"]):
            pct = top.get("pct_score", 0.0)
            reqs = top.get("requirement_scores", [])
            strong = [r["requirement"] for r in reqs if r.get("score", 0) >= 2]
            return (
                f"Based on the evaluation, **{top_name}** is the strongest candidate "
                f"with a {pct:.0%} match score. "
                + (f"They show clear evidence for: {', '.join(strong[:3])}." if strong else "")
            )

        # "Why is X ranked above Y?"
        ranked_re = re.search(r"why is (\w+) ranked above (\w+)", q)
        if ranked_re or "ranked above" in q or "why" in q:
            if len(evaluation_results) >= 2:
                second = evaluation_results[1]
                second_name = second.get("candidate_name", "the second candidate")
                diff = top.get("pct_score", 0) - second.get("pct_score", 0)
                top_strong = [
                    r["requirement"] for r in top.get("requirement_scores", [])
                    if r.get("score", 0) >= 2
                ]
                return (
                    f"{top_name} is ranked above {second_name} by a {diff:.0%} score margin. "
                    f"{top_name}'s strongest areas: {', '.join(top_strong[:3]) or 'see evaluation'}."
                )

        # "Compare candidates" / general comparison
        if any(w in q for w in ["compare", "comparison", "versus", "vs", "differ"]):
            lines = [f"Candidate comparison for this role:"]
            for c in evaluation_results:
                name = c.get("candidate_name", "?")
                pct = c.get("pct_score", 0.0)
                lines.append(f"  • {name}: {pct:.0%} overall match")
            return "\n".join(lines)

        # Generic fallback
        return (
            f"The top-ranked candidate is {top_name} ({top.get('pct_score', 0):.0%} match). "
            "For a more detailed answer, please ensure an LLM backend is configured."
        )
