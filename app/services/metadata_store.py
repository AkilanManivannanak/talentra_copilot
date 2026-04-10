from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import CandidateRecord, DocumentRecord, RecruiterNote, Requirement, RoleRecord


class MetadataStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "metadata.json"
        self._version = 0
        self._state = self._load()

    @property
    def version(self) -> int:
        return self._version

    def _empty_state(self) -> dict:
        return {"roles": [], "candidates": [], "documents": [], "notes": []}

    def _load(self) -> dict:
        if not self._path.exists():
            return self._empty_state()
        data = json.loads(self._path.read_text())
        if not isinstance(data, dict):
            return self._empty_state()
        for key in ["roles", "candidates", "documents", "notes"]:
            data.setdefault(key, [])
        for candidate in data["candidates"]:
            candidate.setdefault("stage", "Applied")
            candidate.setdefault("shortlisted", False)
        self._version += 1
        return data

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2, default=str))
        self._version += 1

    def create_role(self, title: str, description: str, requirements: list[Requirement]) -> RoleRecord:
        role = RoleRecord(
            id=uuid.uuid4().hex[:12],
            title=title,
            description=description,
            requirements=requirements,
            document_ids=[],
            created_at=datetime.now(timezone.utc),
        )
        self._state["roles"].append(role.model_dump(mode="json"))
        self._save()
        return role

    def attach_document_to_role(self, role_id: str, document_id: str) -> RoleRecord:
        for idx, raw in enumerate(self._state["roles"]):
            if raw["id"] == role_id:
                doc_ids = list(raw.get("document_ids", []))
                if document_id not in doc_ids:
                    doc_ids.append(document_id)
                raw["document_ids"] = doc_ids
                self._state["roles"][idx] = raw
                self._save()
                return RoleRecord.model_validate(raw)
        raise KeyError(f"Unknown role_id: {role_id}")

    def create_or_get_candidate(self, name: str) -> CandidateRecord:
        normalised = name.strip().lower()
        for raw in self._state["candidates"]:
            if raw["name"].strip().lower() == normalised:
                raw.setdefault("stage", "Applied")
                raw.setdefault("shortlisted", False)
                return CandidateRecord.model_validate(raw)
        candidate = CandidateRecord(
            id=uuid.uuid4().hex[:12],
            name=name,
            document_ids=[],
            stage="Applied",
            shortlisted=False,
            created_at=datetime.now(timezone.utc),
        )
        self._state["candidates"].append(candidate.model_dump(mode="json"))
        self._save()
        return candidate

    def attach_document_to_candidate(self, candidate_id: str, document_id: str) -> CandidateRecord:
        for idx, raw in enumerate(self._state["candidates"]):
            if raw["id"] == candidate_id:
                doc_ids = list(raw.get("document_ids", []))
                if document_id not in doc_ids:
                    doc_ids.append(document_id)
                raw["document_ids"] = doc_ids
                self._state["candidates"][idx] = raw
                self._save()
                return CandidateRecord.model_validate(raw)
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    def update_candidate_stage(self, candidate_id: str, stage: str) -> CandidateRecord:
        for idx, raw in enumerate(self._state["candidates"]):
            if raw["id"] == candidate_id:
                raw["stage"] = stage
                if stage == "Shortlisted":
                    raw["shortlisted"] = True
                self._state["candidates"][idx] = raw
                self._save()
                return CandidateRecord.model_validate(raw)
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    def update_candidate_shortlist(self, candidate_id: str, shortlisted: bool) -> CandidateRecord:
        for idx, raw in enumerate(self._state["candidates"]):
            if raw["id"] == candidate_id:
                raw["shortlisted"] = shortlisted
                if shortlisted and raw.get("stage") == "Applied":
                    raw["stage"] = "Shortlisted"
                self._state["candidates"][idx] = raw
                self._save()
                return CandidateRecord.model_validate(raw)
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    def register_document(self, *, filename: str, entity_type: str, entity_id: str, entity_name: str, chunk_count: int) -> DocumentRecord:
        document = DocumentRecord(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            chunk_count=chunk_count,
            uploaded_at=datetime.now(timezone.utc),
        )
        self._state["documents"].append(document.model_dump(mode="json"))
        self._save()
        return document

    def update_document_chunk_count(self, document_id: str, chunk_count: int) -> DocumentRecord:
        for idx, raw in enumerate(self._state["documents"]):
            if raw["id"] == document_id:
                raw["chunk_count"] = chunk_count
                self._state["documents"][idx] = raw
                self._save()
                return DocumentRecord.model_validate(raw)
        raise KeyError(f"Unknown document_id: {document_id}")

    def remove_document(self, document_id: str) -> None:
        changed = False
        new_documents = [raw for raw in self._state["documents"] if raw["id"] != document_id]
        if len(new_documents) != len(self._state["documents"]):
            self._state["documents"] = new_documents
            changed = True

        for key in ("roles", "candidates"):
            for idx, raw in enumerate(self._state[key]):
                doc_ids = [doc_id for doc_id in raw.get("document_ids", []) if doc_id != document_id]
                if doc_ids != raw.get("document_ids", []):
                    raw["document_ids"] = doc_ids
                    self._state[key][idx] = raw
                    changed = True
        if changed:
            self._save()

    def remove_candidate_if_orphaned(self, candidate_id: str) -> None:
        new_candidates = []
        changed = False
        for raw in self._state["candidates"]:
            if raw["id"] == candidate_id and not raw.get("document_ids"):
                changed = True
                continue
            new_candidates.append(raw)
        if changed:
            self._state["candidates"] = new_candidates
            self._state["notes"] = [note for note in self._state["notes"] if note.get("candidate_id") != candidate_id]
            self._save()

    def add_note(self, candidate_id: str, text: str) -> RecruiterNote:
        self.get_candidate(candidate_id)
        note = RecruiterNote(
            id=uuid.uuid4().hex[:12],
            candidate_id=candidate_id,
            text=text.strip(),
            created_at=datetime.now(timezone.utc),
        )
        self._state["notes"].append(note.model_dump(mode="json"))
        self._save()
        return note

    def list_notes(self, candidate_id: str) -> list[RecruiterNote]:
        notes = [RecruiterNote.model_validate(raw) for raw in self._state.get("notes", []) if raw.get("candidate_id") == candidate_id]
        notes.sort(key=lambda note: note.created_at, reverse=True)
        return notes

    def count_notes(self, candidate_id: str) -> int:
        return sum(1 for raw in self._state.get("notes", []) if raw.get("candidate_id") == candidate_id)

    def get_role(self, role_id: str) -> RoleRecord:
        for raw in self._state["roles"]:
            if raw["id"] == role_id:
                return RoleRecord.model_validate(raw)
        raise KeyError(f"Unknown role_id: {role_id}")

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        for raw in self._state["candidates"]:
            if raw["id"] == candidate_id:
                raw.setdefault("stage", "Applied")
                raw.setdefault("shortlisted", False)
                return CandidateRecord.model_validate(raw)
        raise KeyError(f"Unknown candidate_id: {candidate_id}")

    def list_roles(self) -> list[RoleRecord]:
        return [RoleRecord.model_validate(item) for item in self._state["roles"]]

    def list_candidates(self) -> list[CandidateRecord]:
        return [CandidateRecord.model_validate(item) for item in self._state["candidates"]]

    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord.model_validate(item) for item in self._state["documents"]]
