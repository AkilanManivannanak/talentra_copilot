from __future__ import annotations
import re

# Heuristic patterns that often precede requirement lists
_SECTION_RE = re.compile(
    r"(?:requirements?|qualifications?|responsibilities|must.have|you.will|we.need)[:\s]*",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^[\s\-\*\•\·\u2022\u2023]+", re.MULTILINE)


def extract_requirements(text: str, max_req: int = 12) -> list[str]:
    """
    Pull requirement bullet points from a JD.
    Returns list of concise requirement strings.
    """
    # Try to isolate the requirements section
    m = _SECTION_RE.search(text)
    body = text[m.end():] if m else text

    lines = body.splitlines()
    reqs = []
    for line in lines:
        clean = _BULLET_RE.sub("", line).strip()
        if 10 < len(clean) < 200:
            reqs.append(clean)
        if len(reqs) >= max_req:
            break

    # Fallback: split on common tech keywords if no bullets found
    if not reqs:
        for kw in re.findall(r"[A-Z][a-zA-Z0-9\+\#\./\- ]{4,50}(?:experience|knowledge|skills?)", text):
            reqs.append(kw.strip())
            if len(reqs) >= max_req:
                break

    # Final fallback: split sentences
    if not reqs:
        for sent in re.split(r"[.;]", text):
            s = sent.strip()
            if 15 < len(s) < 150:
                reqs.append(s)
            if len(reqs) >= max_req:
                break

    return reqs[:max_req]
