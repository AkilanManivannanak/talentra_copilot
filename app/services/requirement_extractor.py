from __future__ import annotations

import re
from typing import Iterable

from app.models.schemas import Requirement

_TECH_KEYWORDS = [
    "python", "fastapi", "rag", "retrieval", "vector database", "vector databases", "embeddings",
    "llm", "llms", "machine learning", "ml", "ai", "nlp", "docker", "kubernetes", "aws", "gcp",
    "azure", "streamlit", "sql", "api", "apis", "evaluation", "prompt engineering", "langchain",
    "faiss", "pytorch", "tensorflow", "fine-tuning", "fine tuning", "agent", "agents",
]


class RequirementExtractor:
    def __init__(self, *, chat_model: str, openai_api_key: str, max_requirements: int) -> None:
        self._max_requirements = max_requirements

    def extract(self, title: str, description: str) -> list[Requirement]:
        requirements = self._extract_fallback(title, description)
        return requirements[: self._max_requirements]

    def _extract_fallback(self, title: str, description: str) -> list[Requirement]:
        lines = [line.strip(" -•\t") for line in description.splitlines() if line.strip()]
        candidates: list[tuple[str, float]] = []
        keyword_pattern = re.compile(
            r"(must|should|required|experience|proficient|knowledge|familiar|build|design|develop|deploy|maintain|python|ml|ai|rag|fastapi|streamlit|docker|sql|aws|gcp|azure)",
            re.IGNORECASE,
        )

        for line in lines:
            if len(line) < 12:
                continue
            if keyword_pattern.search(line):
                weight = 1.2 if re.search(r"must|required|strong", line, re.IGNORECASE) else 1.0
                candidates.append((self._clean_requirement_text(line), weight))

        if len(candidates) < self._max_requirements:
            for skill in self._extract_skill_phrases(description):
                candidates.append((f"Experience with {skill}", 1.0))

        if len(candidates) < self._max_requirements:
            sentence_candidates = re.split(r"(?<=[.!?])\s+", description)
            for sentence in sentence_candidates:
                sentence = sentence.strip()
                if len(sentence) >= 20:
                    candidates.append((self._clean_requirement_text(sentence), 0.9))

        unique = list(self._unique(candidates))[: self._max_requirements]
        return [
            Requirement(id=f"req_{idx}", text=text, weight=max(0.5, min(weight, 2.0)))
            for idx, (text, weight) in enumerate(unique, start=1)
        ]

    def _extract_skill_phrases(self, description: str) -> list[str]:
        found: list[str] = []
        desc = description.lower()
        for keyword in _TECH_KEYWORDS:
            if keyword in desc:
                phrase = keyword.upper() if keyword in {"ai", "ml", "nlp", "rag", "llm", "llms", "sql", "aws", "gcp"} else keyword
                found.append(phrase)
        return found

    def _clean_requirement_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip().rstrip(".")
        text = re.sub(r"^(responsibilities|requirements|qualifications)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        return text

    def _unique(self, items: Iterable[tuple[str, float]]) -> Iterable[tuple[str, float]]:
        seen: set[str] = set()
        for text, weight in items:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            yield text, weight
