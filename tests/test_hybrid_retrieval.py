"""Fusion logic, tested with a deterministic fake embedder.

The dense backends (LSA, sentence-transformers) are exercised end-to-end by
`eval/harness.py`. These tests pin the *fusion* behaviour, which is where the subtle
bugs live and which must not depend on a model being installed.
"""
import tempfile
from pathlib import Path

import pytest

from app.services.hybrid import DenseIndex, HybridRetriever, _rrf_to_unit
from app.services.vectorstore import VectorStoreService


class FakeEmbedder:
    """Maps text to a 3-d vector by counting three marker words, then normalises."""

    requires_corpus_fit = False
    model_name = "fake-3d"
    available = True
    load_error = None

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        raw = [float(lowered.count(word)) for word in ("alpha", "beta", "gamma")]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def encode_documents(self, texts):
        return [self._vector(text) for text in texts]

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture
def retriever_factory():
    created = []

    def build(mode: str):
        tmpdir = tempfile.mkdtemp()
        created.append(tmpdir)
        lexical = VectorStoreService(
            embedding_model="local-lexical",
            vectorstore_path=str(Path(tmpdir) / "index.json"),
            openai_api_key="",
        )
        dense = DenseIndex(Path(tmpdir), FakeEmbedder())
        return HybridRetriever(lexical=lexical, dense=dense, mode=mode, rrf_k=60)

    yield build


def index(retriever, docs):
    for doc_id, text in docs:
        retriever.add_document(
            text=text,
            metadata_base={
                "document_id": doc_id, "entity_type": "candidate", "entity_id": doc_id,
                "entity_name": doc_id, "filename": f"{doc_id}.txt",
            },
        )


DOCS = [
    ("d1", "alpha alpha alpha engineering work with distributed pipelines and services"),
    ("d2", "beta beta engineering work with distributed pipelines and services"),
    ("d3", "gamma engineering work with distributed pipelines and deployment services"),
]


def test_lexical_mode_never_consults_the_dense_index(retriever_factory):
    retriever = retriever_factory("lexical")
    index(retriever, DOCS)
    assert retriever.mode == "lexical"
    hits = retriever.search(query="distributed pipelines", k=3)
    assert hits and all(hit["provenance"]["source"] == "lexical" for hit in hits)


def test_hybrid_marks_provenance_and_fuses_both_lists(retriever_factory):
    retriever = retriever_factory("hybrid")
    index(retriever, DOCS)
    assert retriever.mode == "hybrid"
    hits = retriever.search(query="alpha distributed pipelines", k=3)
    sources = {hit["provenance"]["source"] for hit in hits}
    assert sources & {"both", "dense", "lexical"}
    top = hits[0]
    assert top["provenance"]["fused_score"] > 0
    assert 0.0 <= top["score"] <= 1.0


def test_hybrid_degrades_to_lexical_when_dense_is_empty(retriever_factory):
    retriever = retriever_factory("hybrid")
    # Nothing indexed: the dense index has no records, so hybrid must report lexical.
    assert retriever.mode == "lexical"
    assert retriever.requested_mode == "hybrid"


def test_rrf_maps_into_the_unit_interval():
    # Both retrievers ranking a document first is the ceiling.
    both_first = 2 * (1 / (60 + 1))
    assert _rrf_to_unit(both_first, 60) == pytest.approx(1.0)
    assert _rrf_to_unit(0.0, 60) == 0.0
    assert 0.0 < _rrf_to_unit(1 / 61, 60) < 1.0


def test_search_grouped_returns_one_bucket_per_entity(retriever_factory):
    """Regression: scoring each candidate with its own filtered search made every hit
    rank 1, so RRF gave every candidate the same fused score and hybrid scored *below*
    both of its inputs in the first ablation run."""
    retriever = retriever_factory("hybrid")
    index(retriever, DOCS)
    grouped = retriever.search_grouped(
        query="engineering services", per_entity_k=2, filters={"entity_type": "candidate"}
    )
    assert set(grouped) <= {"d1", "d2", "d3"}
    assert grouped, "grouped search must return buckets"
    for hits in grouped.values():
        assert len(hits) <= 2
    # Distinct documents must not all collapse to an identical score.
    tops = [max(h["score"] for h in hits) for hits in grouped.values()]
    assert len({round(score, 6) for score in tops}) > 1
