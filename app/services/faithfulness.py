"""
Sentence-level groundedness scoring for generated answers.

The Copilot's whole premise is "evidence-grounded", but a citation list next to an
answer proves nothing about whether the citations actually support the sentences.
This module measures that directly:

    faithfulness = supported_sentences / claim_sentences

A sentence counts as supported when at least one cited chunk covers it — measured by
cosine similarity when the dense backend is installed, and by IDF-weighted content-term
coverage otherwise. Both paths report which backend produced the number, because a
metric whose method is invisible is not a metric.

Sentences carrying no content terms (pure connective text) are excluded from the
denominator rather than counted as failures, and the excluded set is reported.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.models.schemas import Evidence, FaithfulnessReport, SentenceGrounding
from app.services.embeddings import LocalEmbedder

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9'\"(])")
_WORD = re.compile(r"[a-zA-Z0-9_+#./-]+")

# Terms that carry no verifiable claim; excluded when deciding whether a sentence
# makes a claim at all, and when measuring lexical coverage.
_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "based", "but", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "they", "this",
    "to", "was", "were", "with", "which", "who", "you", "your", "our", "we", "not", "no", "than",
    "then", "there", "here", "more", "most", "next", "also", "while", "shows", "show", "showing",
    "candidate", "candidates", "role", "results", "result", "evaluation", "requirement",
    "requirements", "scored", "score", "scores", "matched", "match", "matches", "closest",
    "strongest", "stronger", "strong", "evidence", "targeted", "overall", "ranking", "ranked",
    "above", "below", "main", "areas", "coverage", "specific", "topic",
    "so", "if", "yes", "do", "does", "did", "can", "could", "will", "would",
    "should", "may", "might", "must", "been", "being", "am", "all", "any", "some", "such",
}


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in (raw.lower().strip("._-") for raw in _WORD.findall(text))
        if token and token not in _FUNCTION_WORDS and len(token) > 1 and not _is_bare_number(token)
    ]


def _is_bare_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


class FaithfulnessScorer:
    """Scores an answer against its own citations. Never calls out to a network service."""

    def __init__(
        self,
        *,
        embedder: LocalEmbedder | None = None,
        dense_threshold: float = 0.55,
        lexical_threshold: float = 0.5,
    ) -> None:
        self._embedder = embedder
        self._dense_threshold = dense_threshold
        self._lexical_threshold = lexical_threshold

    @property
    def backend(self) -> str:
        return "dense-cosine" if (self._embedder and self._embedder.available) else "lexical-coverage"

    @property
    def threshold(self) -> float:
        return self._dense_threshold if self.backend == "dense-cosine" else self._lexical_threshold

    def score(self, *, answer: str, citations: Sequence[Evidence]) -> FaithfulnessReport:
        sentences = split_sentences(answer)
        claim_sentences = [s for s in sentences if _content_terms(s)]
        threshold = self.threshold

        if not claim_sentences:
            return FaithfulnessReport(
                sentences_total=0,
                sentences_supported=0,
                faithfulness=1.0,
                threshold=threshold,
                unsupported=[],
                detail=[],
            )
        if not citations:
            return FaithfulnessReport(
                sentences_total=len(claim_sentences),
                sentences_supported=0,
                faithfulness=0.0,
                threshold=threshold,
                unsupported=list(claim_sentences),
                detail=[
                    SentenceGrounding(sentence=s, supported=False, best_citation_index=None, support_score=0.0)
                    for s in claim_sentences
                ],
            )

        passages = [f"{c.entity_name}. {c.snippet}" for c in citations]
        matrix = self._support_matrix(claim_sentences, passages)

        detail: list[SentenceGrounding] = []
        supported = 0
        unsupported: list[str] = []
        for sentence, row in zip(claim_sentences, matrix, strict=False):
            best_index = max(range(len(row)), key=lambda i: row[i]) if row else None
            best_score = row[best_index] if best_index is not None else 0.0
            is_supported = best_score >= threshold
            if is_supported:
                supported += 1
            else:
                unsupported.append(sentence)
            detail.append(
                SentenceGrounding(
                    sentence=sentence,
                    supported=is_supported,
                    best_citation_index=best_index if is_supported else None,
                    support_score=round(best_score, 4),
                )
            )

        return FaithfulnessReport(
            sentences_total=len(claim_sentences),
            sentences_supported=supported,
            faithfulness=round(supported / len(claim_sentences), 4),
            threshold=threshold,
            unsupported=unsupported,
            detail=detail,
        )

    # -- backends ----------------------------------------------------------

    def _support_matrix(self, sentences: list[str], passages: list[str]) -> list[list[float]]:
        if self._embedder and self._embedder.available:
            dense = self._dense_matrix(sentences, passages)
            if dense is not None:
                return dense
        return self._lexical_matrix(sentences, passages)

    def _dense_matrix(self, sentences: list[str], passages: list[str]) -> list[list[float]] | None:
        sentence_vectors = self._embedder.encode_documents(sentences)
        passage_vectors = self._embedder.encode_documents(passages)
        if sentence_vectors is None or passage_vectors is None:
            return None
        return [
            [sum(a * b for a, b in zip(sv, pv, strict=False)) for pv in passage_vectors]
            for sv in sentence_vectors
        ]

    def _lexical_matrix(self, sentences: list[str], passages: list[str]) -> list[list[float]]:
        """IDF-weighted recall of a sentence's content terms inside a passage.

        Recall rather than F1: a passage is allowed to say more than the sentence does,
        but every claim term in the sentence has to appear somewhere in the passage.
        Rare terms are weighted up so that matching 'BEVFormer' counts for more than
        matching 'built'.
        """
        passage_terms = [set(_content_terms(passage)) for passage in passages]
        document_frequency: Counter[str] = Counter()
        for terms in passage_terms:
            document_frequency.update(terms)
        total = max(1, len(passages))

        def idf(term: str) -> float:
            return math.log((1 + total) / (1 + document_frequency.get(term, 0))) + 1.0

        matrix: list[list[float]] = []
        for sentence in sentences:
            terms = _content_terms(sentence)
            weights = {term: idf(term) for term in set(terms)}
            denominator = sum(weights.values()) or 1.0
            matrix.append(
                [
                    sum(weight for term, weight in weights.items() if term in bag) / denominator
                    for bag in passage_terms
                ]
            )
        return matrix
