"""Parse work tenure dates and compute years of experience."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

# Match patterns like "Jan 2019 – Mar 2022", "2018-Present", "2020 to 2023"
_RANGE_RE = re.compile(
    r"(?P<start>"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?"
    r"\s*\d{4})"
    r"\s*(?:–|—|-|to|until|through)\s*"
    r"(?P<end>"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?"
    r"\s*(?:\d{4}|present|current|now|ongoing))",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date_str(s: str) -> Optional[date]:
    s = s.strip().lower()
    if s in ("present", "current", "now", "ongoing"):
        return date.today()
    # Try "Month YYYY"
    for abbr, month_num in _MONTH_MAP.items():
        if s.startswith(abbr):
            year_match = re.search(r"\d{4}", s)
            if year_match:
                return date(int(year_match.group()), month_num, 1)
    # Try plain year
    year_match = re.search(r"\d{4}", s)
    if year_match:
        return date(int(year_match.group()), 1, 1)
    return None


def parse_tenure(text: str) -> list[dict]:
    """
    Extract date ranges from text.
    Returns list of dicts: {start, end, duration_years, raw}.
    """
    results = []
    for m in _RANGE_RE.finditer(text):
        start_date = _parse_date_str(m.group("start"))
        end_date = _parse_date_str(m.group("end"))
        if start_date and end_date and end_date >= start_date:
            delta_years = round((end_date - start_date).days / 365.25, 1)
            results.append({
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "duration_years": delta_years,
                "raw": m.group(0),
            })
    return results


def total_years_experience(text: str) -> float:
    """Sum of all non-overlapping tenure durations found in text."""
    tenures = parse_tenure(text)
    if not tenures:
        return 0.0
    # Simple sum (not overlap-aware — good enough for portfolio use)
    return round(sum(t["duration_years"] for t in tenures), 1)
