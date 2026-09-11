"""
Dense embedding backends.

Two are provided, because "dense retrieval" should not be gated behind a model download
that a locked-down network may refuse:

* **LSA** (`EMBEDDING_MODEL=lsa`, the default) — TF-IDF followed by truncated SVD, fitted
  on the indexed corpus. Latent semantic analysis is genuinely dense and genuinely
  captures synonymy, which is the exact failure mode the lexical retriever has: it can
  connect "semantic search over embedded passages" to a query about retrieval-augmented
  generation, where term matching cannot. It needs only scikit-learn from PyPI, downloads
  nothing, and fits in under a second at this corpus size — so the ablation in
  `eval/harness.py` is reproducible by anyone who cloned the repo.

* **Sentence-transformer** (`EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`) — a pretrained
  bi-encoder, ~130 MB, CPU. Better semantics than LSA, at the cost of an optional install
  (`requirements-ml.txt`) and a one-time model download.

Both keep the project's zero-external-API-cost property: everything runs in-process.

`available` is the single honest signal for whether a backend can run. Every caller
checks it; nothing pretends a dense index exists when it does not, and `/ops/build`
reports the backend actually in force.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# bge-* models are trained with an asymmetric prefix for retrieval queries.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Lazily-loaded sentence-transformer. Thread-safe, process-local, no network at query time."""

    requires_corpus_fit = False

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", *, enabled: bool = True) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._model: Any | None = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _load(self) -> Any | None:
        if not self._enabled:
            return None
        with self._lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self._model_name, device="cpu")
                logger.info("Loaded embedding model %s", self._model_name)
            except Exception as exc:  # pragma: no cover - depends on optional install
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._model = None
                logger.warning(
                    "Dense embeddings unavailable (%s). Falling back to lexical retrieval.",
                    self._load_error,
                )
            return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def load_error(self) -> str | None:
        self._load()
        return self._load_error

    @property
    def dimension(self) -> int:
        model = self._load()
        if model is None:
            return 0
        return int(model.get_sentence_embedding_dimension())

    # -- encoding ----------------------------------------------------------

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]] | None:
        model = self._load()
        if model is None or not texts:
            return None
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return [row.tolist() for row in vectors]

    def encode_query(self, text: str) -> list[float] | None:
        model = self._load()
        if model is None or not text.strip():
            return None
        prefixed = f"{_BGE_QUERY_PREFIX}{text}" if "bge" in self._model_name.lower() else text
        vector = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()


class LSAEmbedder:
    """Latent semantic analysis over the indexed corpus. No download, no network.

    Unlike a pretrained encoder this backend is *corpus-fitted*: the projection is learned
    from the documents being indexed, so adding documents invalidates it. `requires_corpus_fit`
    tells the index to refit and re-encode rather than append. That is O(corpus) per ingest,
    which is the honest trade for needing no pretrained weights — fine for hundreds of
    resumes, and the reason the sentence-transformer backend exists for anything larger.
    """

    requires_corpus_fit = True

    def __init__(self, *, dimensions: int = 192, enabled: bool = True) -> None:
        self._dimensions = dimensions
        self._enabled = enabled
        self._vectorizer: Any | None = None
        self._svd: Any | None = None
        self._fitted = False
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_name(self) -> str:
        return f"lsa-tfidf-svd-{self._dimensions}"

    @property
    def available(self) -> bool:
        if not self._enabled:
            return False
        try:
            import sklearn  # type: ignore  # noqa: F401

            return True
        except ImportError as exc:  # pragma: no cover
            self._load_error = f"scikit-learn not installed: {exc}"
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def dimension(self) -> int:
        return self._dimensions if self._fitted else 0

    def fit(self, corpus: Sequence[str]) -> bool:
        """Refit the projection. Returns False when the corpus is too small to be useful."""
        if not self.available or len(corpus) < 2:
            return False
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        with self._lock:
            # sublinear_tf dampens repeated terms; min_df=1 because resumes are short and
            # a term appearing in one document is exactly the signal we want to keep.
            self._vectorizer = TfidfVectorizer(
                lowercase=True, sublinear_tf=True, min_df=1, stop_words="english", ngram_range=(1, 2)
            )
            matrix = self._vectorizer.fit_transform(list(corpus))
            components = max(2, min(self._dimensions, matrix.shape[1] - 1, len(corpus) - 1))
            self._svd = TruncatedSVD(n_components=components, random_state=0)
            self._svd.fit(matrix)
            self._fitted = True
            return True

    def _project(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not self._fitted or self._vectorizer is None or self._svd is None:
            return None
        import numpy as np  # type: ignore

        projected = self._svd.transform(self._vectorizer.transform(list(texts)))
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (projected / norms).tolist()

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]] | None:
        return self._project(texts) if texts else None

    def encode_query(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        vectors = self._project([text])
        return vectors[0] if vectors else None


def build_embedder(model_name: str, *, enabled: bool) -> LocalEmbedder | LSAEmbedder:
    """Pick a dense backend from the configured model name.

    `lsa` (or any `lsa-*`) selects the corpus-fitted backend; anything else is treated as
    a sentence-transformers model id.
    """
    if model_name.lower().startswith("lsa"):
        return LSAEmbedder(enabled=enabled)
    return LocalEmbedder(model_name, enabled=enabled)


class LocalReranker:
    """Optional cross-encoder reranking stage. Same optional-install contract as LocalEmbedder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", *, enabled: bool = False) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._model: Any | None = None
        self._load_attempted = False
        self._lock = threading.Lock()

    def _load(self) -> Any | None:
        if not self._enabled:
            return None
        with self._lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self._model_name, device="cpu", max_length=512)
                logger.info("Loaded reranker %s", self._model_name)
            except Exception as exc:  # pragma: no cover - depends on optional install
                self._model = None
                logger.warning("Reranker unavailable (%s: %s).", type(exc).__name__, exc)
            return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, passages: Sequence[str]) -> list[float] | None:
        model = self._load()
        if model is None or not passages:
            return None
        scores = model.predict([(query, passage) for passage in passages], show_progress_bar=False)
        return [float(value) for value in scores]
