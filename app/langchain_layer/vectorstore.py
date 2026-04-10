"""
Vector store abstraction.
Priority: Chroma (persistent) → FAISS (in-memory) → legacy lexical index.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _get_embeddings():
    """Embeddings disabled — using lexical backend."""
    return None
# ---------------------------------------------------------------------------
# Lexical (IDF-style) fallback — original Talentra algorithm
# ---------------------------------------------------------------------------

def _idf_score(query_terms: list[str], doc: str) -> float:
    doc_lower = doc.lower()
    score = 0.0
    for term in query_terms:
        tf = doc_lower.count(term.lower())
        if tf:
            score += math.log(1 + tf) * (1 + 0.5 * (len(term) > 5))
    return score


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TalentraVectorStore:
    """
    Unified vector store that wraps Chroma, FAISS, or the legacy lexical index.
    Instantiate once per vectorstore_path; add_documents() and search() are the
    only public surface.
    """

    def __init__(self, path: str | Path, backend: str = "lexical"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._backend: str = "lexical"
        self._store: Any = None
        self._docs: list[dict] = []  # lexical fallback store
        self._embeddings = None

        if backend == "auto":
            self._init_auto()
        elif backend == "chroma":
            self._init_chroma()
        elif backend == "faiss":
            self._init_faiss()
        else:
            self._load_lexical()

    def _init_auto(self):
        try:
            self._init_chroma()
        except Exception:
            try:
                self._init_faiss()
            except Exception:
                self._load_lexical()

    def _init_chroma(self):
        import chromadb  # type: ignore
        from langchain_community.vectorstores import Chroma  # type: ignore
        self._embeddings = _get_embeddings()
        if self._embeddings is None:
            raise ImportError("No embeddings available for Chroma")
        client = chromadb.PersistentClient(path=str(self.path / "chroma"))
        self._store = Chroma(
            client=client,
            collection_name="talentra",
            embedding_function=self._embeddings,
        )
        self._backend = "chroma"

    def _init_faiss(self):
        import faiss  # type: ignore  # noqa: F401
        from langchain_community.vectorstores import FAISS  # type: ignore
        self._embeddings = _get_embeddings()
        if self._embeddings is None:
            raise ImportError("No embeddings available for FAISS")
        faiss_index = self.path / "faiss.index"
        if faiss_index.exists():
            self._store = FAISS.load_local(
                str(self.path), self._embeddings, allow_dangerous_deserialization=True
            )
        else:
            # Will be created on first add_documents
            self._store = None
        self._backend = "faiss"

    def _load_lexical(self):
        self._backend = "lexical"
        index_path = self.path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                self._docs = json.load(f)

    def _save_lexical(self):
        index_path = self.path / "index.json"
        with open(index_path, "w") as f:
            json.dump(self._docs, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[dict], doc_id: str) -> None:
        """
        chunks: list of {text, metadata} dicts from splitter.
        doc_id: candidate or role id for filtering.
        """
        if self._backend == "chroma":
            from langchain.schema import Document  # type: ignore
            lc_docs = [
                Document(page_content=c["text"], metadata={"doc_id": doc_id, **c.get("metadata", {})})
                for c in chunks
            ]
            self._store.add_documents(lc_docs)

        elif self._backend == "faiss":
            from langchain.schema import Document  # type: ignore
            from langchain_community.vectorstores import FAISS  # type: ignore
            lc_docs = [
                Document(page_content=c["text"], metadata={"doc_id": doc_id, **c.get("metadata", {})})
                for c in chunks
            ]
            if self._store is None:
                self._store = FAISS.from_documents(lc_docs, self._embeddings)
            else:
                self._store.add_documents(lc_docs)
            self._store.save_local(str(self.path))

        else:  # lexical
            for c in chunks:
                self._docs.append({
                    "doc_id": doc_id,
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                })
            self._save_lexical()

    def search(self, query: str, doc_id: str | None = None, top_k: int = 5) -> list[dict]:
        """
        Returns list of {text, score, metadata, doc_id}.
        Optionally filter by doc_id.
        """
        if self._backend in ("chroma", "faiss") and self._store is not None:
            filter_dict = {"doc_id": doc_id} if doc_id else None
            try:
                kwargs = {"k": top_k}
                if filter_dict:
                    kwargs["filter"] = filter_dict
                results = self._store.similarity_search_with_score(query, **kwargs)
                return [
                    {
                        "text": doc.page_content,
                        "score": float(score),
                        "metadata": doc.metadata,
                        "doc_id": doc.metadata.get("doc_id", ""),
                    }
                    for doc, score in results
                ]
            except Exception:
                pass  # fall through to lexical

        # Lexical
        candidates = [d for d in self._docs if doc_id is None or d["doc_id"] == doc_id]
        query_terms = re.findall(r"\w+", query.lower())
        scored = [(d, _idf_score(query_terms, d["text"])) for d in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": d["text"], "score": s, "metadata": d.get("metadata", {}), "doc_id": d["doc_id"]}
            for d, s in scored[:top_k]
            if s > 0
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks for a given doc_id."""
        if self._backend == "lexical":
            self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
            self._save_lexical()
        # Chroma / FAISS deletion by metadata filter is version-dependent; skip for now

    @property
    def backend(self) -> str:
        return self._backend
