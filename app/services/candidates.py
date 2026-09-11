from __future__ import annotations

import logging

from app.models.schemas import CandidateUploadItem
from app.preprocessing import run_preprocessing_pipeline
from app.services.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class CandidateService:
    """Resume ingest.

    The preprocessing pipeline (clean -> sections -> skills -> PII redaction -> tenure)
    used to exist but was only reachable from an unwired graph module, so no uploaded
    resume ever passed through it. It now runs on every upload, and — importantly — the
    text that reaches the index is the *redacted* text when redaction is enabled, so
    contact details are removed before they can ever be retrieved as evidence.
    """

    def __init__(
        self,
        *,
        metadata: MetadataStore,
        vectorstore,
        redact_pii: bool = True,
        use_spacy: bool = True,
    ) -> None:
        self._metadata = metadata
        self._vectorstore = vectorstore
        self._redact_pii = redact_pii
        self._use_spacy = use_spacy

    def ingest_candidate_resume(self, candidate_name: str, filename: str, text: str) -> CandidateUploadItem:
        processed = run_preprocessing_pipeline(
            text,
            redact=self._redact_pii,
            use_spacy=self._use_spacy,
            metadata={"filename": filename},
        )
        indexable_text = processed.redacted_text if self._redact_pii else processed.cleaned_text
        if not indexable_text.strip():
            raise ValueError(f"{filename} produced no indexable text after preprocessing.")

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
                text=indexable_text,
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
            enriched = self._metadata.enrich_candidate(
                candidate.id,
                skills=processed.skills,
                pii_types_redacted=processed.pii_types_found,
                total_years_experience=processed.total_years_experience,
            )
        except Exception:
            self._metadata.remove_document(document.id)
            self._metadata.remove_candidate_if_orphaned(candidate.id)
            raise

        return CandidateUploadItem(
            id=candidate.id,
            name=candidate.name,
            filename=filename,
            chunk_count=chunk_count,
            skills=enriched.skills,
            pii_types_redacted=enriched.pii_types_redacted,
        )
