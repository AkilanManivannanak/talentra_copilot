"""
Hybrid retrieval: lexical (IDF + phrase) fused with dense (bi-encoder) using
Reciprocal Rank Fusion, with an optional cross-encoder reranking stage.

Why RRF rather than score interpolation: the two retrievers produce scores on
incomparable scales (bounded heuristic vs cosine similarity), so a weighted sum
needs per-corpus calibration that would silently rot. RRF only consumes ranks,
which makes the fusion parameter-light and stable across corpora.

    rrf(d) = sum over retrievers of  1 / (k + rank_r(d))

The retriever presents exactly the same `search()` contract as VectorStoreService,
so RankingService and CopilotService are unaware of which mode is active. That is
what makes the ablation in `eval/harness.py` a fair comparison: only the retriever
changes between runs.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.services.embeddings import LocalReranker
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: Any) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        os.replace(tmp_name, path)
    except Exception:
        if Path(tmp_name).exists():
            os.unlink(tmp_name)
        raise


class DenseIndex:
    """Flat cosine index over unit-normalised vectors. Exact search, no ANN structure.

    At portfolio scale (hundreds of chunks) a flat scan is faster than building and
    maintaining an ANN graph, and it removes an entire class of recall bug. The
    interface is deliberately the shape an ANN backend would need, so swapping in
    FAISS later is a one-class change.
    """

    def __init__(self, path: str | Path, embedder: Any) -> None:
        self._path = Path(path)
        self._embedder = embedder
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._version = 0
        self._dirty = False
        self._load()

    @property
    def _storage_file(self) -> Path:
        base = self._path
        if base.suffix:
            base = base.parent
        return base / "dense_index.json"

    @property
    def version(self) -> int:
        return self._version

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def ready(self) -> bool:
        """Usable for search: backend available and at least one record present.

        Vectors may not be materialised yet for a corpus-fitted backend — see
        `_ensure_fitted`, which encodes lazily on first search.
        """
        return self._embedder.available and bool(self._records)

    def _load(self) -> None:
        file_path = self._storage_file
        if not file_path.exists():
            return
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("model") == self._embedder.model_name:
                self._records = payload.get("records", [])
                self._version += 1
            elif isinstance(payload, dict):
                logger.warning(
                    "Dense index was built with %s but configured model is %s; discarding index.",
                    payload.get("model"),
                    self._embedder.model_name,
                )
        except Exception as exc:
            logger.warning("Could not read dense index (%s); starting empty.", exc)
            self._records = []

    def _save(self) -> None:
        _atomic_write_json(
            self._storage_file,
            {"model": self._embedder.model_name, "records": self._records},
        )
        self._version += 1

    def add(self, chunks: Sequence[dict[str, Any]]) -> int:
        """chunks: [{content, metadata}] — metadata must carry chunk_id."""
        if not chunks or not self._embedder.available:
            return 0

        if getattr(self._embedder, "requires_corpus_fit", False):
            return self._add_with_refit(chunks)

        vectors = self._embedder.encode_documents([chunk["content"] for chunk in chunks])
        if vectors is None:
            return 0
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=False):
                self._records.append(
                    {"content": chunk["content"], "metadata": chunk["metadata"], "vector": vector}
                )
            self._save()
        return len(vectors)

    def _add_with_refit(self, chunks: Sequence[dict[str, Any]]) -> int:
        """Corpus-fitted backends (LSA) must relearn the projection when the corpus grows,
        then re-encode everything — vectors from an older projection are not comparable to
        vectors from a newer one, and mixing them silently corrupts the ranking.

        The refit is deferred rather than run here. Doing it per document turned a
        12-resume batch upload into 12 full refits: 10.7s, and quadratic in corpus size.
        Marking the index dirty and refitting once on the next search makes a batch upload
        one refit regardless of batch size.
        """
        with self._lock:
            for chunk in chunks:
                self._records.append(
                    {"content": chunk["content"], "metadata": chunk["metadata"], "vector": []}
                )
            self._dirty = True
            self._version += 1
        return len(chunks)

    def _ensure_fitted(self) -> bool:
        """Refit and re-encode if documents were added since the last search."""
        if not self._dirty:
            return True
        with self._lock:
            if not self._dirty:
                return True
            corpus = [record["content"] for record in self._records]
            if not self._embedder.fit(corpus):
                return False
            vectors = self._embedder.encode_documents(corpus)
            if vectors is None:
                return False
            for record, vector in zip(self._records, vectors, strict=False):
                record["vector"] = vector
            self._dirty = False
            self._save()
        return True

    def search(
        self,
        *,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._records or not self._embedder.available:
            return []
        if getattr(self._embedder, "requires_corpus_fit", False) and not self._ensure_fitted():
            return []
        query_vector = self._embedder.encode_query(query)
        if query_vector is None:
            return []

        eligible = [
            record
            for record in self._records
            if not filters
            or all(record.get("metadata", {}).get(key) == value for key, value in filters.items())
        ]
        if not eligible:
            return []

        scored = [
            {
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _cosine(query_vector, record["vector"]),
            }
            for record in eligible
            if record.get("vector")
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Both sides are unit-normalised at encode time, so the dot product is the cosine."""
    try:
        import numpy as np  # type: ignore

        return float(np.dot(np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")))
    except Exception:
        return float(sum(x * y for x, y in zip(a, b, strict=False)))


class HybridRetriever:
    """Drop-in replacement for VectorStoreService.search with lexical/dense/hybrid modes."""

    def __init__(
        self,
        *,
        lexical: VectorStoreService,
        dense: DenseIndex | None,
        reranker: LocalReranker | None = None,
        mode: str = "lexical",
        rrf_k: int = 60,
        dense_candidates: int = 50,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self._reranker = reranker
        self._requested_mode = mode
        self._rrf_k = max(1, rrf_k)
        self._dense_candidates = max(1, dense_candidates)

    # -- introspection -----------------------------------------------------

    @property
    def mode(self) -> str:
        """The mode actually in force, which may be narrower than the one requested."""
        if self._requested_mode == "lexical":
            return "lexical"
        if self._dense is None or self._dense.size == 0 or not self._dense.ready:
            return "lexical"
        return self._requested_mode

    @property
    def requested_mode(self) -> str:
        return self._requested_mode

    @property
    def version(self) -> int:
        dense_version = self._dense.version if self._dense else 0
        return self._lexical.version * 1000 + dense_version

    @property
    def rerank_active(self) -> bool:
        return bool(self._reranker and self._reranker.available)

    # -- ingest ------------------------------------------------------------

    def add_document(self, *, text: str, metadata_base: dict[str, Any]) -> int:
        """Writes to the lexical index, and mirrors the accepted chunks into the dense index."""
        count = self._lexical.add_document(text=text, metadata_base=metadata_base)
        if self._dense is not None and self._requested_mode != "lexical":
            accepted = self._lexical.records_for_document(metadata_base["document_id"])
            self._dense.add(accepted)
        return count

    # -- retrieval ---------------------------------------------------------

    def search(
        self,
        *,
        query: str,
        k: int = 5,
        filters: dict[str, str] | None = None,
        fetch_k: int = 30,
    ) -> list[dict[str, Any]]:
        mode = self.mode
        lexical_hits = (
            self._lexical.search(query=query, k=k, filters=filters, fetch_k=fetch_k)
            if mode in ("lexical", "hybrid")
            else []
        )
        if mode == "lexical":
            for hit in lexical_hits:
                hit.setdefault("provenance", {"source": "lexical"})
            return lexical_hits[:k]

        dense_hits = self._dense.search(
            query=query, k=self._dense_candidates, filters=filters
        ) if self._dense else []

        if mode == "dense":
            results = [
                {
                    "content": hit["content"],
                    "snippet": self._lexical.make_snippet(hit["content"], query),
                    "metadata": hit["metadata"],
                    "score": max(0.0, min(float(hit["score"]), 1.0)),
                    "provenance": {
                        "source": "dense",
                        "dense_rank": rank,
                        "dense_score": float(hit["score"]),
                    },
                }
                for rank, hit in enumerate(dense_hits, start=1)
            ]
            return self._maybe_rerank(query, results, k)

        fused = self._fuse(lexical_hits, dense_hits, query)
        return self._maybe_rerank(query, fused, k)

    def _fuse(
        self,
        lexical_hits: list[dict[str, Any]],
        dense_hits: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        def key_of(hit: dict[str, Any]) -> str:
            metadata = hit.get("metadata", {})
            return str(metadata.get("chunk_id") or hash(hit.get("content", "")))

        for rank, hit in enumerate(lexical_hits, start=1):
            entry = merged.setdefault(
                key_of(hit),
                {"content": hit["content"], "metadata": hit["metadata"], "provenance": {}, "rrf": 0.0},
            )
            entry["snippet"] = hit.get("snippet")
            entry["rrf"] += 1.0 / (self._rrf_k + rank)
            entry["provenance"].update({"lexical_rank": rank, "lexical_score": float(hit["score"])})

        for rank, hit in enumerate(dense_hits, start=1):
            entry = merged.setdefault(
                key_of(hit),
                {"content": hit["content"], "metadata": hit["metadata"], "provenance": {}, "rrf": 0.0},
            )
            entry["rrf"] += 1.0 / (self._rrf_k + rank)
            entry["provenance"].update({"dense_rank": rank, "dense_score": float(hit["score"])})

        results: list[dict[str, Any]] = []
        for entry in merged.values():
            provenance = entry["provenance"]
            has_lexical = "lexical_rank" in provenance
            has_dense = "dense_rank" in provenance
            provenance["source"] = "both" if has_lexical and has_dense else ("lexical" if has_lexical else "dense")
            provenance["fused_score"] = entry["rrf"]
            results.append(
                {
                    "content": entry["content"],
                    "snippet": entry.get("snippet") or self._lexical.make_snippet(entry["content"], query),
                    "metadata": entry["metadata"],
                    # RRF scores are tiny by construction; map to a 0-1 band so downstream
                    # thresholds (match_threshold) keep the same meaning across modes.
                    "score": _rrf_to_unit(entry["rrf"], self._rrf_k),
                    "provenance": provenance,
                }
            )
        results.sort(key=lambda item: item["provenance"]["fused_score"], reverse=True)
        return results

    def search_grouped(
        self,
        *,
        query: str,
        per_entity_k: int = 2,
        filters: dict[str, str] | None = None,
        fetch_k: int = 200,
    ) -> dict[str, list[dict[str, Any]]]:
        """One corpus-wide search, results grouped by entity_id.

        This exists because scoring candidates with one filtered search *each* is both
        slower and, for rank-fusion, wrong: with a single document in the eligible set
        every hit is rank 1, so RRF assigns every candidate the same fused score and the
        ranking collapses. That is exactly what the first ablation run showed — hybrid
        scored *below* both of its inputs (nDCG 0.73 vs 0.78 lexical, 0.80 dense).

        Retrieving over the whole pool once and grouping afterwards gives ranks their
        meaning back, makes IDF global rather than per-document, and turns N searches per
        requirement into one.
        """
        hits = self.search(query=query, k=fetch_k, filters=filters, fetch_k=fetch_k)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            entity_id = hit.get("metadata", {}).get("entity_id")
            if entity_id is None:
                continue
            bucket = grouped.setdefault(entity_id, [])
            if len(bucket) < per_entity_k:
                bucket.append(hit)
        return grouped

    def _maybe_rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        if not results:
            return []
        if not self.rerank_active:
            return results[:k]
        window = results[: max(k * 4, 20)]
        scores = self._reranker.score(query, [item["content"] for item in window])
        if scores is None:
            return results[:k]
        for item, score in zip(window, scores, strict=False):
            item["provenance"]["rerank_score"] = score
            item["score"] = _sigmoid(score)
        window.sort(key=lambda item: item["provenance"]["rerank_score"], reverse=True)
        return window[:k]


def _rrf_to_unit(rrf_score: float, rrf_k: int) -> float:
    """Map an RRF score onto 0-1. Two retrievers both ranking a chunk first is the ceiling."""
    ceiling = 2.0 / (rrf_k + 1)
    return max(0.0, min(rrf_score / ceiling, 1.0))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
