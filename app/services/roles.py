from __future__ import annotations

from app.models.schemas import RoleRecord
from app.services.metadata_store import MetadataStore
from app.services.requirement_extractor import RequirementExtractor
from app.services.vectorstore import VectorStoreService


class RoleService:
    def __init__(
        self,
        *,
        metadata: MetadataStore,
        vectorstore: VectorStoreService,
        requirement_extractor: RequirementExtractor,
    ) -> None:
        self._metadata = metadata
        self._vectorstore = vectorstore
        self._extractor = requirement_extractor

    def create_role_from_text(self, title: str, description: str) -> RoleRecord:
        requirements = self._extractor.extract(title, description)
        return self._metadata.create_role(title=title, description=description, requirements=requirements)

    def create_role_from_document(self, title: str, filename: str, text: str) -> RoleRecord:
        role = self.create_role_from_text(title, text)
        document = self._metadata.register_document(
            filename=filename,
            entity_type="role",
            entity_id=role.id,
            entity_name=role.title,
            chunk_count=0,
        )
        try:
            chunk_count = self._vectorstore.add_document(
                text=text,
                metadata_base={
                    "document_id": document.id,
                    "entity_type": "role",
                    "entity_id": role.id,
                    "entity_name": role.title,
                    "filename": filename,
                },
            )
            self._metadata.update_document_chunk_count(document.id, chunk_count)
            return self._metadata.attach_document_to_role(role.id, document.id)
        except Exception:
            self._metadata.remove_document(document.id)
            raise
