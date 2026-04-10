"""PII redaction using Microsoft Presidio with regex fallback."""
from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Regex fallbacks (always available)
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("email",    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    ("phone",    re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)"), "<PHONE>"),
    ("ssn",      re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "<SSN>"),
    ("zip_us",   re.compile(r"\b\d{5}(?:-\d{4})?\b"), "<ZIP>"),
    ("linkedin", re.compile(r"linkedin\.com/in/[a-zA-Z0-9\-_/]+"), "<LINKEDIN>"),
    ("github",   re.compile(r"github\.com/[a-zA-Z0-9\-_/]+"), "<GITHUB>"),
    ("url",      re.compile(r"https?://[^\s]+"), "<URL>"),
]


@lru_cache(maxsize=1)
def _load_presidio():
    """Try to load Presidio AnalyzerEngine; return None if not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        engine = AnalyzerEngine()
        return engine
    except Exception:
        return None


def redact_pii(text: str, language: str = "en") -> tuple[str, list[str]]:
    """
    Redact PII from text.
    Returns (redacted_text, list_of_pii_types_found).
    Tries Presidio first; falls back to regex patterns.
    """
    redacted = text
    found_types: list[str] = []

    engine = _load_presidio()
    if engine is not None:
        try:
            results = engine.analyze(text=text, language=language)
            # Sort by position descending so offsets stay valid after replacement
            results_sorted = sorted(results, key=lambda r: r.start, reverse=True)
            for result in results_sorted:
                placeholder = f"<{result.entity_type}>"
                redacted = redacted[: result.start] + placeholder + redacted[result.end :]
                found_types.append(result.entity_type)
            return redacted, list(set(found_types))
        except Exception:
            pass  # fall through to regex

    # Regex fallback
    for label, pattern, placeholder in _PII_PATTERNS:
        if pattern.search(redacted):
            found_types.append(label.upper())
            redacted = pattern.sub(placeholder, redacted)

    return redacted, list(set(found_types))


def is_clean(text: str) -> bool:
    """Quick check: does text still contain obvious PII?"""
    _, types = redact_pii(text)
    return len(types) == 0
