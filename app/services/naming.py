from __future__ import annotations

import re
from pathlib import Path


class NamingService:
    def derive_candidate_name(self, filename: str, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            first = lines[0]
            if 2 <= len(first.split()) <= 4 and not re.search(r"[@|:]", first):
                return first
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
        return stem.title() or "Unknown Candidate"

    def derive_role_title(self, filename: str, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and len(lines[0]) <= 80:
            return lines[0]
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
        return stem.title() or "Untitled Role"
