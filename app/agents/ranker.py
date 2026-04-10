"""RankerAgent: evidence-weighted, requirement-level candidate scoring."""
from __future__ import annotations

import math
import re
from typing import Any

from .base import BaseAgent


class RankerAgent(BaseAgent):
    """
    Scores candidates per requirement using vectorstore retrieval,
    then aggregates into a ranked list with evidence quotes.
    Mirrors (and improves) the original evaluation service.
    """

    SCORE_LABELS = {0: "no evidence", 1: "weak", 2: "clear", 3: "strong"}

    def __init__(self, llm: Any | None = None, vectorstore: Any | None = None):
        super().__init__(llm)
        self._vectorstore = vectorstore

    def rank(
        self,
        role_id: str,
        role_title: str,
        requirements: list[str],
        candidate_ids: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Returns candidates sorted descending by aggregate score.
        Each entry: {candidate_id, candidate_name, total_score, max_score,
                     pct_score, requirement_scores: [{req, score, evidence}]}
        """
        scored_candidates = []

        for cid in candidate_ids:
            req_scores = []
            for req in requirements:
                score, evidence = self._score_requirement(
                    candidate_id=cid,
                    requirement=req,
                    top_k=top_k,
                )
                req_scores.append({
                    "requirement": req,
                    "score": score,
                    "score_label": self.SCORE_LABELS.get(score, "unknown"),
                    "evidence": evidence,
                })

            total = sum(r["score"] for r in req_scores)
            max_possible = 3 * len(requirements) if requirements else 1
            pct = round(total / max_possible, 3) if max_possible else 0.0

            # Derive a display name from candidate_id (e.g. "akila_resume.txt" → "Akila")
            name = self._id_to_name(cid)

            scored_candidates.append({
                "candidate_id": cid,
                "candidate_name": name,
                "total_score": total,
                "max_score": max_possible,
                "pct_score": pct,
                "requirement_scores": req_scores,
            })

        scored_candidates.sort(key=lambda x: x["pct_score"], reverse=True)
        return scored_candidates

    def _score_requirement(
        self,
        candidate_id: str,
        requirement: str,
        top_k: int = 5,
    ) -> tuple[int, str]:
        """
        Retrieve top_k chunks for (candidate, requirement) and return (score 0-3, best_evidence).
        If LLM available, delegates scoring to it; otherwise uses lexical heuristic.
        """
        chunks = []
        if self._vectorstore is not None:
            try:
                chunks = self._vectorstore.search(
                    query=requirement, doc_id=candidate_id, top_k=top_k
                )
            except Exception:
                pass

        if not chunks:
            return 0, ""

        best_chunk = chunks[0]
        evidence_text = best_chunk.get("text", "")

        if self._llm:
            return self._llm_score(requirement, evidence_text)

        return self._lexical_score(requirement, [c.get("text", "") for c in chunks])

    def _lexical_score(self, requirement: str, texts: list[str]) -> tuple[int, str]:
        """Heuristic 0-3 score based on term coverage."""
        req_terms = re.findall(r"\w+", requirement.lower())
        req_terms = [t for t in req_terms if len(t) > 2]

        best_score = 0
        best_text = ""
        for text in texts:
            raw = self._score_evidence_lexical(req_terms, text)
            # Map raw IDF score to 0-3 scale
            if raw <= 0:
                sc = 0
            elif raw < 2:
                sc = 1
            elif raw < 5:
                sc = 2
            else:
                sc = 3
            if sc > best_score:
                best_score = sc
                best_text = text

        return best_score, best_text[:300]

    def _llm_score(self, requirement: str, evidence: str) -> tuple[int, str]:
        """Ask LLM to score evidence for requirement (0-3)."""
        from app.langchain_layer.prompts import EVALUATION_PROMPT, format_prompt
        prompt = format_prompt(
            EVALUATION_PROMPT,
            role_title="",
            requirement=requirement,
            evidence=evidence[:800],
        )
        reply = self._call_llm(prompt, max_tokens=200)
        parsed = self._parse_json(reply)
        if isinstance(parsed, dict):
            score = int(parsed.get("score", 0))
            quote = parsed.get("evidence_quote", evidence[:200])
            return min(max(score, 0), 3), quote
        return self._lexical_score(requirement, [evidence])

    @staticmethod
    def _id_to_name(candidate_id: str) -> str:
        """Extract a human-readable name from a candidate_id / filename."""
        name = re.sub(r"[_\-]resume.*$", "", candidate_id, flags=re.IGNORECASE)
        name = re.sub(r"\.(txt|pdf|docx|md)$", "", name, flags=re.IGNORECASE)
        return name.replace("_", " ").replace("-", " ").title().strip() or candidate_id
