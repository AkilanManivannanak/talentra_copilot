from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.ats import ATSService
from app.services.bias_audit import CounterfactualBiasAuditor
from app.services.candidates import CandidateService
from app.services.copilot import CopilotService
from app.services.embeddings import LocalReranker, build_embedder
from app.services.faithfulness import FaithfulnessScorer
from app.services.hybrid import DenseIndex, HybridRetriever
from app.services.metadata_store import MetadataStore
from app.services.naming import NamingService
from app.services.ranking import RankingService
from app.services.requirement_extractor import RequirementExtractor
from app.services.roles import RoleService
from app.services.summary import SummaryService
from app.services.vectorstore import VectorStoreService


@dataclass
class ServiceContainer:
    settings: Settings
    metadata: MetadataStore
    lexical: VectorStoreService
    embedder: Any
    reranker: LocalReranker
    retriever: HybridRetriever
    naming: NamingService
    requirements: RequirementExtractor
    summary: SummaryService
    faithfulness: FaithfulnessScorer
    roles: RoleService
    candidates: CandidateService
    ranking: RankingService
    copilot: CopilotService
    ats: ATSService
    bias_auditor: CounterfactualBiasAuditor

    # Kept so existing callers that said `services.vectorstore` still work; the
    # retriever is the object everything should be talking to.
    @property
    def vectorstore(self) -> HybridRetriever:
        return self.retriever

    @classmethod
    def from_settings(cls, settings: Settings) -> ServiceContainer:
        metadata = MetadataStore(settings.data_path)
        lexical = VectorStoreService(
            embedding_model="local-lexical",
            vectorstore_path=settings.vectorstore_path,
            openai_api_key="",
            cache_size=settings.search_cache_size,
        )

        dense_wanted = settings.retrieval_mode in ("hybrid", "dense")
        embedder = build_embedder(settings.embedding_model, enabled=dense_wanted)
        reranker = LocalReranker(settings.rerank_model, enabled=settings.rerank_enabled)
        dense_index = DenseIndex(Path(settings.vectorstore_path), embedder) if dense_wanted else None

        retriever = HybridRetriever(
            lexical=lexical,
            dense=dense_index,
            reranker=reranker,
            mode=settings.retrieval_mode,
            rrf_k=settings.rrf_k,
            dense_candidates=settings.dense_candidates,
        )

        naming = NamingService()
        requirements = RequirementExtractor(max_requirements=settings.max_requirements)
        summary = SummaryService()
        faithfulness = FaithfulnessScorer(embedder=embedder if dense_wanted else None)

        roles = RoleService(metadata=metadata, vectorstore=retriever, requirement_extractor=requirements)
        candidates = CandidateService(
            metadata=metadata,
            vectorstore=retriever,
            redact_pii=settings.redact_pii_on_ingest,
            use_spacy=settings.use_spacy,
        )
        ranking = RankingService(
            settings=settings, metadata=metadata, vectorstore=retriever, summary_service=summary
        )
        copilot = CopilotService(
            metadata=metadata,
            vectorstore=retriever,
            summary_service=summary,
            ranking_service=ranking,
            faithfulness=faithfulness,
        )
        ats = ATSService(metadata=metadata, ranking=ranking)
        bias_auditor = CounterfactualBiasAuditor(
            vectorstore=lexical, match_threshold=settings.match_threshold
        )

        return cls(
            settings=settings,
            metadata=metadata,
            lexical=lexical,
            embedder=embedder,
            reranker=reranker,
            retriever=retriever,
            naming=naming,
            requirements=requirements,
            summary=summary,
            faithfulness=faithfulness,
            roles=roles,
            candidates=candidates,
            ranking=ranking,
            copilot=copilot,
            ats=ats,
            bias_auditor=bias_auditor,
        )
