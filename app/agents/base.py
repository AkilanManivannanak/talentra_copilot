"""Base agent class shared by all Talentra agents."""
from __future__ import annotations

import json
import re
from typing import Any


class BaseAgent:
    """
    Shared base for all Talentra agents.
    Provides: _call_llm(), _parse_json(), _format_evidence().
    Subclasses implement their specific logic.
    """

    def __init__(self, llm: Any | None = None):
        self._llm = llm  # Optional LangChain LLM; agents work without one.

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Call the configured LLM or return empty string.
        Supports LangChain LLM, callable, or None (rule-based fallback).
        """
        if self._llm is None:
            return ""
        try:
            if hasattr(self._llm, "invoke"):
                result = self._llm.invoke(prompt)
                return result.content if hasattr(result, "content") else str(result)
            if callable(self._llm):
                return str(self._llm(prompt))
        except Exception:
            pass
        return ""

    def _parse_json(self, text: str) -> dict | list | None:
        """Extract the first JSON object or array from text."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # Try finding a JSON object/array inline
        match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return None

    def _format_evidence(self, chunks: list[dict], max_chars: int = 1200) -> str:
        """Format retrieval chunks into a readable evidence string."""
        lines = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            snippet = chunk.get("text", "").strip()
            if total + len(snippet) > max_chars:
                snippet = snippet[: max_chars - total]
            lines.append(f"[{i}] {snippet}")
            total += len(snippet)
            if total >= max_chars:
                break
        return "\n".join(lines)

    def _score_evidence_lexical(self, query_terms: list[str], text: str) -> float:
        """IDF-weighted lexical score."""
        import math
        text_lower = text.lower()
        score = 0.0
        for term in query_terms:
            tf = text_lower.count(term.lower())
            if tf:
                score += math.log(1 + tf) * (1 + 0.5 * (len(term) > 5))
        # Penalize very short or contact-noisy chunks
        if len(text) < 40 or re.search(r"@|\d{3}[-.\s]\d{3}", text):
            score *= 0.3
        return score
