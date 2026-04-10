"""Resume and JD section detection using regex heuristics."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Ordered list of (section_label, list_of_keywords_that_trigger_it)
SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("CONTACT",      ["contact", "personal info", "personal details"]),
    ("SUMMARY",      ["summary", "objective", "profile", "about me", "overview"]),
    ("EXPERIENCE",   ["experience", "work history", "employment", "career", "positions held"]),
    ("EDUCATION",    ["education", "academic", "qualifications", "degrees", "schooling"]),
    ("SKILLS",       ["skills", "technical skills", "competencies", "technologies", "tools", "expertise"]),
    ("CERTIFICATIONS", ["certif", "accreditation", "license"]),
    ("PROJECTS",     ["projects", "portfolio", "case studies", "selected work"]),
    ("PUBLICATIONS", ["publication", "papers", "research", "patents"]),
    ("LANGUAGES",    ["languages", "spoken"]),
    ("AWARDS",       ["awards", "honors", "achievements", "recognition"]),
    ("REQUIREMENTS", ["requirements", "qualifications required", "must have", "responsibilities"]),
]

# Build a compiled pattern for each section
_COMPILED: list[tuple[str, re.Pattern]] = [
    (label, re.compile(r"(?i)^\s*(" + "|".join(re.escape(kw) for kw in kws) + r")\s*[:\-]?\s*$", re.MULTILINE))
    for label, kws in SECTION_PATTERNS
]


@dataclass
class Section:
    label: str
    start: int        # char offset in original text
    end: int          # char offset (exclusive)
    text: str


def detect_sections(text: str) -> list[Section]:
    """
    Split text into labelled sections.
    Returns a list of Section objects ordered by appearance.
    """
    # Collect all matches across all patterns
    hits: list[tuple[int, int, str]] = []  # (start, end, label)
    for label, pattern in _COMPILED:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.end(), label))

    if not hits:
        return [Section(label="FULL", start=0, end=len(text), text=text)]

    # Sort by position, remove overlaps (keep first-seen label at each position)
    hits.sort(key=lambda h: h[0])
    clean_hits: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, label in hits:
        if start >= last_end:
            clean_hits.append((start, end, label))
            last_end = end

    # Build sections between header matches
    sections: list[Section] = []
    for i, (hstart, hend, label) in enumerate(clean_hits):
        next_start = clean_hits[i + 1][0] if i + 1 < len(clean_hits) else len(text)
        body = text[hend:next_start].strip()
        sections.append(Section(label=label, start=hstart, end=next_start, text=body))

    return sections


def sections_to_dict(text: str) -> dict[str, str]:
    """Convenience: {label: body_text} — last writer wins for duplicate labels."""
    return {s.label: s.text for s in detect_sections(text)}
