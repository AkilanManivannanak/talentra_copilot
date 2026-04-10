"""Unified preprocessing pipeline: clean → detect sections → extract skills → redact PII → parse dates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cleaner import clean_text, deduplicate_chunks
from .sections import detect_sections, Section
from .skills import extract_skills
from .pii import redact_pii
from .dates import parse_tenure, total_years_experience


@dataclass
class PreprocessedDocument:
    """Output of the full preprocessing pipeline."""
    raw_text: str
    cleaned_text: str
    redacted_text: str
    pii_types_found: list[str]
    sections: list[Section]
    skills: list[str]
    tenures: list[dict]
    total_years_experience: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cleaned_text": self.cleaned_text,
            "redacted_text": self.redacted_text,
            "pii_types_found": self.pii_types_found,
            "sections": [
                {"label": s.label, "text": s.text[:500]} for s in self.sections
            ],
            "skills": self.skills,
            "tenures": self.tenures,
            "total_years_experience": self.total_years_experience,
            "metadata": self.metadata,
        }


def run_preprocessing_pipeline(
    raw_text: str,
    redact: bool = True,
    use_spacy: bool = True,
    metadata: dict | None = None,
) -> PreprocessedDocument:
    """
    Full pipeline:
      1. Clean / normalize text
      2. Detect sections
      3. Extract skills (spaCy + regex)
      4. Redact PII
      5. Parse tenure dates
    """
    # Step 1: clean
    cleaned = clean_text(raw_text)

    # Step 2: sections
    sections = detect_sections(cleaned)

    # Step 3: skills (run on cleaned text before PII redaction so names don't confuse NER)
    skills = extract_skills(cleaned, use_spacy=use_spacy)

    # Step 4: PII redaction
    if redact:
        redacted, pii_types = redact_pii(cleaned)
    else:
        redacted, pii_types = cleaned, []

    # Step 5: tenure
    tenures = parse_tenure(cleaned)
    total_yoe = total_years_experience(cleaned)

    return PreprocessedDocument(
        raw_text=raw_text,
        cleaned_text=cleaned,
        redacted_text=redacted,
        pii_types_found=pii_types,
        sections=sections,
        skills=skills,
        tenures=tenures,
        total_years_experience=total_yoe,
        metadata=metadata or {},
    )
