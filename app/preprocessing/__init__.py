"""Preprocessing pipeline for resume and JD text."""
from .cleaner import clean_text
from .dates import parse_tenure
from .pii import redact_pii
from .pipeline import run_preprocessing_pipeline
from .sections import detect_sections
from .skills import extract_skills


def warm_models() -> dict[str, float]:
    """Build the spaCy pipeline and Presidio engine ahead of the first request.

    Presidio's AnalyzerEngine construction measured 7.8s on a cold process; spaCy adds
    ~0.6s. Paid lazily, that whole cost lands on whoever uploads the first resume, which
    is what made batch upload look like a 10s operation. Both are cached for the process
    lifetime, so warming once at startup removes it from the request path entirely.

    Returns per-model warm-up cost in milliseconds so /ops and the benchmark can report it.
    """
    import time

    timings: dict[str, float] = {}
    sample = "Jane Doe, engineer. jane@example.com. Built Python services. Jan 2020 - Present."

    started = time.perf_counter()
    extract_skills(sample)
    timings["spacy_ms"] = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    redact_pii(sample)
    timings["presidio_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return timings

__all__ = [
    "clean_text",
    "detect_sections",
    "extract_skills",
    "redact_pii",
    "parse_tenure",
    "run_preprocessing_pipeline",
    "warm_models",
]
