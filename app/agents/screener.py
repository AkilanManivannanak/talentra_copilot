"""ScreenerAgent: hard-filter candidates against must-have requirements."""
from __future__ import annotations

import re
from typing import Any

from .base import BaseAgent


class ScreenerAgent(BaseAgent):
    """
    First-pass filter that eliminates candidates who clearly lack must-have
    requirements. Runs before the expensive RankerAgent.
    """

    # Keywords that signal a must-have in a JD requirement string
    MUST_HAVE_SIGNALS = [
        "required", "must have", "must-have", "mandatory", "essential",
        "minimum", "at least", "years of experience", "bachelor", "degree",
    ]

    def __init__(self, llm: Any | None = None, min_score_threshold: float = 0.0):
        super().__init__(llm)
        self.min_score_threshold = min_score_threshold

    def is_must_have(self, requirement: str) -> bool:
        req_lower = requirement.lower()
        return any(sig in req_lower for sig in self.MUST_HAVE_SIGNALS)

    def screen(
        self,
        candidates: list[dict],  # [{id, name, text, skills, ...}]
        requirements: list[str],
        vectorstore: Any | None = None,
    ) -> list[dict]:
        """
        Returns list of candidates with screening results added.
        Each candidate gets: {pass: bool, fail_reasons: [str], screener_notes: str}
        """
        must_haves = [r for r in requirements if self.is_must_have(r)]
        if not must_haves:
            # No hard filters — all candidates pass
            for c in candidates:
                c["screen_pass"] = True
                c["fail_reasons"] = []
                c["screener_notes"] = "No must-have requirements found; all candidates proceed."
            return candidates

        results = []
        for candidate in candidates:
            fail_reasons = []
            candidate_text = (candidate.get("text", "") + " " +
                              " ".join(candidate.get("skills", []))).lower()

            for req in must_haves:
                # Extract key terms from the requirement
                key_terms = self._extract_key_terms(req)
                matched = any(term in candidate_text for term in key_terms)

                # If LLM available, use it for ambiguous cases
                if not matched and self._llm:
                    prompt = (
                        f"Requirement: '{req}'\n"
                        f"Candidate text: '{candidate_text[:800]}'\n"
                        f"Does the candidate meet this requirement? Reply only 'yes' or 'no'."
                    )
                    reply = self._call_llm(prompt, max_tokens=10).strip().lower()
                    matched = reply.startswith("yes")

                if not matched:
                    fail_reasons.append(f"Missing: {req}")

            candidate["screen_pass"] = len(fail_reasons) == 0
            candidate["fail_reasons"] = fail_reasons
            candidate["screener_notes"] = (
                "Passed all must-have checks." if not fail_reasons
                else f"Failed {len(fail_reasons)} must-have(s)."
            )
            results.append(candidate)

        return results

    def _extract_key_terms(self, requirement: str) -> list[str]:
        """Pull meaningful n-grams from a requirement string."""
        stop = {"and", "or", "the", "a", "an", "of", "in", "with", "for",
                "experience", "knowledge", "skills", "ability", "strong"}
        tokens = re.findall(r"[a-zA-Z0-9\+\#\.\/\-]+", requirement.lower())
        terms = [t for t in tokens if t not in stop and len(t) > 1]
        # Also produce 2-grams
        bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms) - 1)]
        return terms + bigrams
