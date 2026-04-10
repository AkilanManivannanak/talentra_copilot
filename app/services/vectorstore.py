from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.document_parser import chunk_text, normalise_whitespace

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]+")
_EMAILISH_RE = re.compile(r"@|\.com$|\.edu$|\.org$|\.net$", re.IGNORECASE)
_PHONEISH_RE = re.compile(r"^\+?\d[\d().\-]{5,}\d$")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "to", "with", "we", "you", "your", "our", "this", "will", "can", "should", "must", "have",
    "has", "had", "their", "they", "them", "about", "into", "than", "then", "also", "not", "such", "using",
    "email", "linkedin", "github", "portfolio", "phone", "mobile", "address", "street", "www", "http", "https",
    "com", "edu", "org", "net", "resume", "curriculum", "vitae", "page",
    "experience", "experiences", "skill", "skills", "knowledge", "familiarity", "required", "preferred",
    "strong", "ability", "abilities",
}
_GENERIC_QUERY_TERMS = {"candidate", "job", "role", "position", "requirement", "requirements", "stronger", "strongest", "best", "results", "evaluation", "experience", "skill", "skills"}
_SIGNAL_TERMS = {"python", "machine", "learn", "ml", "ai", "rag", "retrieval", "fastapi", "docker", "sql", "api", "project", "experience", "internship", "model", "data", "backend", "cloud"}


class VectorStoreService:
    """Local lexical retrieval with aggressive noise control and query-result caching."""

    def __init__(self, *, embedding_model: str, vectorstore_path: str, openai_api_key: str, cache_size: int = 512) -> None:
        self._path = Path(vectorstore_path)
        self._embedding_model = embedding_model
        self._records: list[dict[str, Any]] = []
        self._version = 0
        self._search_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._cache_size = max(32, cache_size)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def configured(self) -> bool:
        return True

    @property
    def version(self) -> int:
        return self._version

    @property
    def _storage_file(self) -> Path:
        if self._path.suffix:
            return self._path
        return self._path / "index.json"

    def _invalidate_cache(self) -> None:
        self._search_cache.clear()

    def _load(self) -> None:
        file_path = self._storage_file
        if not file_path.exists():
            self._records = []
            return
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            self._records = payload if isinstance(payload, list) else []
            self._version += 1
        except Exception:
            self._records = []
        self._invalidate_cache()

    def _save(self) -> None:
        file_path = self._storage_file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        self._version += 1
        self._invalidate_cache()

    def _normalise_token(self, token: str) -> str:
        token = token.strip("._-").lower()
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        return token

    def _is_low_value_token(self, token: str) -> bool:
        if not token or token in _STOPWORDS or len(token) <= 1:
            return True
        if token.isdigit() and len(token) >= 4:
            return True
        if _EMAILISH_RE.search(token) or _PHONEISH_RE.match(token):
            return True
        if token.startswith("http") or token.startswith("www"):
            return True
        return False

    def _tokenise(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in _TOKEN_RE.findall(text.lower()):
            token = self._normalise_token(raw)
            if self._is_low_value_token(token):
                continue
            tokens.append(token)
        return tokens

    def _idf_map(self, docs: list[dict[str, Any]]) -> dict[str, float]:
        if not docs:
            return {}
        df: Counter[str] = Counter()
        for doc in docs:
            for token in set(doc.get("tokens", [])):
                df[token] += 1
        total = len(docs)
        return {token: math.log((1 + total) / (1 + freq)) + 1.0 for token, freq in df.items()}

    def _contact_noise_penalty(self, content: str) -> float:
        lower = content.lower()
        marker_count = 0
        for marker in ("@", "linkedin", "github", "http", "www", ".com", ".edu", "phone", "mobile"):
            if marker in lower:
                marker_count += 1
        if marker_count >= 3:
            return 0.18
        if marker_count >= 2:
            return 0.45
        return 1.0

    def _is_low_signal_chunk(self, content: str, tokens: list[str]) -> bool:
        if not tokens:
            return True
        unique = set(tokens)
        if len(unique) < 5 and self._contact_noise_penalty(content) < 0.5:
            return True
        alpha_terms = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
        if alpha_terms < 6:
            return True
        return False

    def _section_signal_bonus(self, content: str, overlap: set[str]) -> float:
        lower = content.lower()
        bonus = 0.0
        if any(marker in lower for marker in ("experience", "project", "intern", "built", "developed", "skills", "deployed", "production")):
            bonus += 0.05
        if any(token in _SIGNAL_TERMS for token in overlap):
            bonus += 0.07
        return bonus

    def _score_record(self, *, query_tokens: list[str], query_text: str, record: dict[str, Any], idf: dict[str, float]) -> float:
        if not query_tokens:
            return 0.0
        record_tokens = record.get("tokens", [])
        if not record_tokens:
            return 0.0

        tf = Counter(record_tokens)
        query_counts = Counter(query_tokens)

        dot = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        for token, count in query_counts.items():
            weight = idf.get(token, 1.0)
            q_weight = count * weight
            d_weight = tf.get(token, 0) * weight
            dot += q_weight * d_weight
            query_norm += q_weight * q_weight
        for token, count in tf.items():
            weight = idf.get(token, 1.0)
            d_weight = count * weight
            doc_norm += d_weight * d_weight

        lexical = dot / ((math.sqrt(query_norm) * math.sqrt(doc_norm)) + 1e-9)

        content_lower = record["content"].lower()
        phrase_bonus = 0.0
        normalised_query = normalise_whitespace(query_text).lower()
        if normalised_query and len(normalised_query.split()) >= 2 and normalised_query in content_lower:
            phrase_bonus += 0.18

        bigrams = list(zip(query_tokens, query_tokens[1:]))
        if bigrams:
            joined_tokens = " ".join(record_tokens)
            bigram_hits = sum(1 for first, second in bigrams if f"{first} {second}" in joined_tokens)
            phrase_bonus += min(0.18, 0.05 * bigram_hits)

        overlap = set(query_tokens) & set(record_tokens)
        meaningful_overlap = {token for token in overlap if token not in _GENERIC_QUERY_TERMS}
        if not meaningful_overlap:
            return 0.0
        coverage_bonus = min(0.30, len(meaningful_overlap) * 0.05)

        score = (1.8 * lexical) + phrase_bonus + coverage_bonus + self._section_signal_bonus(record["content"], meaningful_overlap)
        score *= self._contact_noise_penalty(record["content"])
        return max(0.0, min(score, 1.0))

    def _make_snippet(self, content: str, query_tokens: list[str]) -> str:
        text = normalise_whitespace(content)
        if not text:
            return ""
        lower = text.lower()
        focus_index = len(text)
        for token in query_tokens:
            if token in _GENERIC_QUERY_TERMS:
                continue
            pos = lower.find(token)
            if pos != -1 and pos < focus_index:
                focus_index = pos
        if focus_index == len(text):
            focus_index = 0
        start = max(0, focus_index - 70)
        end = min(len(text), focus_index + 230)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet

    def add_document(self, *, text: str, metadata_base: dict[str, Any]) -> int:
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document produced no readable text.")

        pending_records: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            tokens = self._tokenise(chunk)
            if self._is_low_signal_chunk(chunk, tokens):
                continue
            metadata = {**metadata_base, "chunk_id": f"{metadata_base['document_id']}:{idx}"}
            pending_records.append({"content": chunk, "metadata": metadata, "tokens": tokens})
        if not pending_records:
            raise ValueError("Document produced only low-signal or contact-only chunks.")
        self._records.extend(pending_records)
        self._save()
        return len(pending_records)

    def search(self, *, query: str, k: int = 5, filters: dict[str, str] | None = None, fetch_k: int = 30) -> list[dict[str, Any]]:
        query_tokens = self._tokenise(query)
        if not query_tokens:
            return []

        filter_items = tuple(sorted((filters or {}).items()))
        cache_key = (self._version, normalise_whitespace(query).lower(), k, fetch_k, filter_items)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        eligible: list[dict[str, Any]] = []
        for record in self._records:
            metadata = record.get("metadata", {})
            if filters and any(metadata.get(key) != value for key, value in filters.items()):
                continue
            eligible.append(record)
        if not eligible:
            return []

        idf = self._idf_map(eligible)
        scored: list[dict[str, Any]] = []
        seen_snippets: set[str] = set()
        for record in eligible:
            score = self._score_record(query_tokens=query_tokens, query_text=query, record=record, idf=idf)
            if score <= 0:
                continue
            snippet = self._make_snippet(record["content"], query_tokens)
            dedupe_key = normalise_whitespace(snippet).lower() or normalise_whitespace(record["content"]).lower()
            if dedupe_key in seen_snippets:
                continue
            seen_snippets.add(dedupe_key)
            scored.append({
                "content": record["content"],
                "snippet": snippet,
                "metadata": record["metadata"],
                "score": score,
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        limit = max(k, min(fetch_k, len(scored))) if fetch_k > k else k
        result = scored[:limit]
        if len(self._search_cache) >= self._cache_size:
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[cache_key] = [dict(item) for item in result]
        return result
