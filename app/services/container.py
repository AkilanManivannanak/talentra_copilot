from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.services.ats import ATSService
from app.services.candidates import CandidateService
from app.services.copilot import CopilotService
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
    vectorstore: VectorStoreService
    naming: NamingService
    requirements: RequirementExtractor
    summary: SummaryService
    roles: RoleService
    candidates: CandidateService
    ranking: RankingService
    copilot: CopilotService
    ats: ATSService

    @classmethod
    def from_settings(cls, settings: Settings) -> "ServiceContainer":
        metadata = MetadataStore(settings.data_path)
        vectorstore = VectorStoreService(
            embedding_model=settings.embedding_model,
            vectorstore_path=settings.vectorstore_path,
            openai_api_key=settings.openai_api_key,
            cache_size=settings.search_cache_size,
        )
        naming = NamingService()
        requirements = RequirementExtractor(
            chat_model=settings.chat_model,
            openai_api_key=settings.openai_api_key,
            max_requirements=settings.max_requirements,
        )
        summary = SummaryService(chat_model=settings.chat_model, openai_api_key=settings.openai_api_key)
        roles = RoleService(metadata=metadata, vectorstore=vectorstore, requirement_extractor=requirements)
        candidates = CandidateService(metadata=metadata, vectorstore=vectorstore)
        ranking = RankingService(settings=settings, metadata=metadata, vectorstore=vectorstore, summary_service=summary)
        copilot = CopilotService(metadata=metadata, vectorstore=vectorstore, summary_service=summary, ranking_service=ranking)
        ats = ATSService(metadata=metadata, ranking=ranking)
        return cls(
            settings=settings,
            metadata=metadata,
            vectorstore=vectorstore,
            naming=naming,
            requirements=requirements,
            summary=summary,
            roles=roles,
            candidates=candidates,
            ranking=ranking,
            copilot=copilot,
            ats=ats,
        )
