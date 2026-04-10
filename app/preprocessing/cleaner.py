"""Text normalization and cleaning utilities."""
from __future__ import annotations

import re
import unicodedata


# Common ligature / smart-quote replacements
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
}

# Patterns that indicate header noise common in parsed PDFs
_NOISE_RE = re.compile(
    r"(page\s+\d+\s+of\s+\d+|confidential|curriculum vitae|resume|cv)\s*",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize, deduplicate whitespace, and remove common PDF artefacts."""
    if not text:
        return ""

    # 1. Ligatures and smart quotes
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)

    # 2. Unicode normalization (NFKC handles ligature forms and width)
    text = unicodedata.normalize("NFKC", text)

    # 3. Strip non-printable control characters (keep \n and \t)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")

    # 4. Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse horizontal whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # 6. Remove common header/footer noise
    text = _NOISE_RE.sub("", text)

    return text.strip()


def deduplicate_chunks(chunks: list[str], sim_threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate chunks using simple character-level overlap."""
    unique: list[str] = []
    for chunk in chunks:
        chunk_set = set(chunk.lower().split())
        is_dup = False
        for kept in unique:
            kept_set = set(kept.lower().split())
            if not chunk_set or not kept_set:
                continue
            overlap = len(chunk_set & kept_set) / max(len(chunk_set), len(kept_set))
            if overlap >= sim_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
    return unique
