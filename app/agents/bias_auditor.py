"""BiasAuditorAgent: detect potential bias patterns in evaluation results."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .base import BaseAgent

# Name-gender association heuristics (illustrative, not exhaustive)
_TYPICALLY_FEMALE = {"akila", "esha", "priya", "sarah", "emma", "olivia", "aisha", "fatima",
                     "maria", "anna", "nina", "sofia", "lily", "grace", "julia"}
_TYPICALLY_MALE   = {"jaxon", "james", "john", "michael", "robert", "william", "david",
                     "daniel", "liam", "noah", "ethan", "mason", "lucas", "oliver"}

# Top-brand school names (prestige bias signal)
_PRESTIGE_SCHOOLS = {"mit", "stanford", "harvard", "oxford", "cambridge", "princeton",
                     "yale", "columbia", "caltech", "carnegie mellon", "cmu"}

# Keywords suggesting recency bias risk (career gaps)
_GAP_SIGNALS = re.compile(
    r"(career break|sabbatical|family leave|gap year|hiatus|took time off)",
    re.IGNORECASE,
)


class BiasAuditorAgent(BaseAgent):
    """
    Reviews ranked candidate list for common hiring bias patterns:
      1. Gender-correlated score patterns
      2. Educational prestige bias
      3. Recency / career-gap bias
      4. Name-based ordering effects

    Returns structured flags and severity level.
    """

    def __init__(self, llm: Any | None = None):
        super().__init__(llm)

    def audit(
        self,
        ranked_candidates: list[dict],
        candidate_texts: dict[str, str] | None = None,
    ) -> dict:
        """
        ranked_candidates: list of RankerAgent output dicts (ordered, highest first).
        candidate_texts: {candidate_id: raw_text} for deeper analysis.
        Returns: {bias_flags, severity, recommendation, detail}
        """
        flags = []
        candidate_texts = candidate_texts or {}

        # --- 1. Gender-correlated score pattern ---
        gender_scores: dict[str, list[float]] = defaultdict(list)
        for c in ranked_candidates:
            name = c.get("candidate_name", "").lower().split()[0]
            pct = c.get("pct_score", 0.0)
            if name in _TYPICALLY_FEMALE:
                gender_scores["female"].append(pct)
            elif name in _TYPICALLY_MALE:
                gender_scores["male"].append(pct)

        if gender_scores.get("female") and gender_scores.get("male"):
            avg_f = sum(gender_scores["female"]) / len(gender_scores["female"])
            avg_m = sum(gender_scores["male"]) / len(gender_scores["male"])
            gap = abs(avg_f - avg_m)
            if gap > 0.25:
                direction = "male" if avg_m > avg_f else "female"
                flags.append(
                    f"Gender score gap detected: {direction}-presenting names score "
                    f"{gap:.0%} higher on average. Review requirement scoring for neutrality."
                )

        # --- 2. Prestige school bias ---
        for c in ranked_candidates:
            text = candidate_texts.get(c.get("candidate_id", ""), "").lower()
            if any(school in text for school in _PRESTIGE_SCHOOLS):
                top_ranked_schools = [
                    cand for cand in ranked_candidates[:2]
                    if any(s in candidate_texts.get(cand.get("candidate_id", ""), "").lower()
                           for s in _PRESTIGE_SCHOOLS)
                ]
                if top_ranked_schools:
                    flags.append(
                        "Top-ranked candidates include prestige-school graduates. "
                        "Verify scoring is driven by demonstrated skills, not school name."
                    )
                break

        # --- 3. Recency / career-gap bias ---
        for c in ranked_candidates:
            text = candidate_texts.get(c.get("candidate_id", ""), "")
            if _GAP_SIGNALS.search(text):
                rank_pos = ranked_candidates.index(c) + 1
                if rank_pos == len(ranked_candidates):
                    flags.append(
                        f"{c.get('candidate_name', 'A candidate')} appears to have a career gap "
                        f"and is ranked last. Confirm gap is not penalizing score unfairly."
                    )

        # --- 4. LLM deep audit (if available) ---
        if self._llm and ranked_candidates:
            llm_flags = self._llm_audit(ranked_candidates)
            flags.extend(llm_flags)

        # Deduplicate flags
        flags = list(dict.fromkeys(flags))

        severity = "none" if not flags else ("low" if len(flags) == 1 else
                                              "medium" if len(flags) <= 3 else "high")

        recommendation = {
            "none": "No bias patterns detected. Proceed with confidence.",
            "low":  "Minor signals found. Review flagged items before final decision.",
            "medium": "Multiple bias signals. Conduct blind review of flagged candidates.",
            "high": "Significant bias risk. Pause and conduct structured review.",
        }[severity]

        return {
            "bias_flags": flags,
            "severity": severity,
            "recommendation": recommendation,
            "candidates_audited": len(ranked_candidates),
        }

    def _llm_audit(self, ranked_candidates: list[dict]) -> list[str]:
        from app.langchain_layer.prompts import BIAS_AUDIT_PROMPT, format_prompt
        summary = "\n".join(
            f"{c['candidate_name']}: {c['pct_score']:.0%}" for c in ranked_candidates
        )
        prompt = format_prompt(BIAS_AUDIT_PROMPT, candidate_scores=summary)
        reply = self._call_llm(prompt, max_tokens=400)
        parsed = self._parse_json(reply)
        if isinstance(parsed, dict):
            return parsed.get("bias_flags", [])
        return []
