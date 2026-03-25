from __future__ import annotations

from app.models.schemas import CandidateUploadItem
from app.services.metadata_store import MetadataStore
from app.services.vectorstore import VectorStoreService


class CandidateService:
    def __init__(self, *, metadata: MetadataStore, vectorstore: VectorStoreService) -> None:
        self._metadata = metadata
        self._vectorstore = vectorstore

    def ingest_candidate_resume(self, candidate_name: str, filename: str, text: str) -> CandidateUploadItem:
        candidate = self._metadata.create_or_get_candidate(candidate_name)
        document = self._metadata.register_document(
            filename=filename,
            entity_type="candidate",
            entity_id=candidate.id,
            entity_name=candidate.name,
            chunk_count=0,
        )
        try:
            chunk_count = self._vectorstore.add_document(
                text=text,
                metadata_base={
                    "document_id": document.id,
                    "entity_type": "candidate",
                    "entity_id": candidate.id,
                    "entity_name": candidate.name,
                    "filename": filename,
                },
            )
            self._metadata.update_document_chunk_count(document.id, chunk_count)
            self._metadata.attach_document_to_candidate(candidate.id, document.id)
        except Exception:
            self._metadata.remove_document(document.id)
            self._metadata.remove_candidate_if_orphaned(candidate.id)
            raise

        return CandidateUploadItem(id=candidate.id, name=candidate.name, filename=filename, chunk_count=chunk_count)
