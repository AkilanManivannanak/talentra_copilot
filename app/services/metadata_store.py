from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.models.schemas import CandidateRecord, DocumentRecord, RecruiterNote, Requirement, RoleRecord


class MetadataStore:
    """JSON-backed metadata store.

    Three properties this needs and previously lacked:

    * **Atomicity** — every mutation used to be a full `write_text` of the whole state,
      so a crash mid-write truncated the file. Writes now go to a temp file in the same
      directory and land via `os.replace`, which is atomic on POSIX and NTFS.
    * **Mutual exclusion** — read-modify-write with no lock loses one of two concurrent
      uploads under uvicorn's threadpool. All mutations now hold a re-entrant lock.
    * **Constant-time lookup** — `get_candidate` walked the whole list. On-disk shape is
      unchanged (a list, for readability and backwards compatibility) but an id index is
      maintained in memory alongside it.

    JSON rather than a database is a deliberate choice for a local-first, zero-infra
    portfolio app: the store is inspectable with `cat`. The trade-off is documented in
    the README rather than hidden behind the phrase "production-grade".
    """

    _COLLECTIONS = ("roles", "candidates", "documents", "notes")

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "metadata.json"
        self._version = 0
        self._lock = threading.RLock()
        self._state = self._load()
        self._index: dict[str, dict[str, dict]] = {}
        self._reindex()

    # -- lifecycle ---------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    def _empty_state(self) -> dict:
        return {key: [] for key in self._COLLECTIONS}

    def _load(self) -> dict:
        if not self._path.exists():
            return self._empty_state()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self._empty_state()
        if not isinstance(data, dict):
            return self._empty_state()
        for key in self._COLLECTIONS:
            data.setdefault(key, [])
        for candidate in data["candidates"]:
            candidate.setdefault("stage", "Applied")
            candidate.setdefault("shortlisted", False)
            candidate.setdefault("skills", [])
            candidate.setdefault("pii_types_redacted", [])
            candidate.setdefault("total_years_experience", 0.0)
        self._version += 1
        return data

    def _reindex(self) -> None:
        self._index = {
            key: {row["id"]: row for row in self._state.get(key, []) if "id" in row}
            for key in ("roles", "candidates", "documents")
        }

    def _save(self) -> None:
        """Atomic replace — never leaves a partially written store on disk."""
        handle, tmp_name = tempfile.mkstemp(dir=str(self._root), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._state, stream, indent=2, default=str)
            os.replace(tmp_name, self._path)
        except Exception:
            if Path(tmp_name).exists():
                os.unlink(tmp_name)
            raise
        self._version += 1

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    # -- roles -------------------------------------------------------------

    def create_role(self, title: str, description: str, requirements: list[Requirement]) -> RoleRecord:
        role = RoleRecord(
            id=self._new_id(),
            title=title,
            description=description,
            requirements=requirements,
            document_ids=[],
            created_at=self._now(),
        )
        with self._lock:
            raw = role.model_dump(mode="json")
            self._state["roles"].append(raw)
            self._index["roles"][role.id] = raw
            self._save()
        return role

    def attach_document_to_role(self, role_id: str, document_id: str) -> RoleRecord:
        with self._lock:
            raw = self._index["roles"].get(role_id)
            if raw is None:
                raise KeyError(f"Unknown role_id: {role_id}")
            doc_ids = list(raw.get("document_ids", []))
            if document_id not in doc_ids:
                doc_ids.append(document_id)
            raw["document_ids"] = doc_ids
            self._save()
            return RoleRecord.model_validate(raw)

    def get_role(self, role_id: str) -> RoleRecord:
        raw = self._index["roles"].get(role_id)
        if raw is None:
            raise KeyError(f"Unknown role_id: {role_id}")
        return RoleRecord.model_validate(raw)

    def list_roles(self) -> list[RoleRecord]:
        return [RoleRecord.model_validate(item) for item in self._state["roles"]]

    # -- candidates --------------------------------------------------------

    def create_or_get_candidate(self, name: str) -> CandidateRecord:
        normalised = name.strip().lower()
        with self._lock:
            for raw in self._state["candidates"]:
                if raw["name"].strip().lower() == normalised:
                    return CandidateRecord.model_validate(self._with_defaults(raw))
            candidate = CandidateRecord(
                id=self._new_id(),
                name=name,
                document_ids=[],
                stage="Applied",
                shortlisted=False,
                created_at=self._now(),
            )
            raw = candidate.model_dump(mode="json")
            self._state["candidates"].append(raw)
            self._index["candidates"][candidate.id] = raw
            self._save()
            return candidate

    def enrich_candidate(
        self,
        candidate_id: str,
        *,
        skills: list[str] | None = None,
        pii_types_redacted: list[str] | None = None,
        total_years_experience: float | None = None,
    ) -> CandidateRecord:
        """Attach preprocessing output (skills, PII types found, tenure) to a candidate."""
        with self._lock:
            raw = self._index["candidates"].get(candidate_id)
            if raw is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            if skills is not None:
                raw["skills"] = sorted(set(raw.get("skills", [])) | set(skills))
            if pii_types_redacted is not None:
                raw["pii_types_redacted"] = sorted(set(raw.get("pii_types_redacted", [])) | set(pii_types_redacted))
            if total_years_experience is not None:
                raw["total_years_experience"] = max(raw.get("total_years_experience", 0.0), total_years_experience)
            self._save()
            return CandidateRecord.model_validate(self._with_defaults(raw))

    def attach_document_to_candidate(self, candidate_id: str, document_id: str) -> CandidateRecord:
        with self._lock:
            raw = self._index["candidates"].get(candidate_id)
            if raw is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            doc_ids = list(raw.get("document_ids", []))
            if document_id not in doc_ids:
                doc_ids.append(document_id)
            raw["document_ids"] = doc_ids
            self._save()
            return CandidateRecord.model_validate(self._with_defaults(raw))

    def update_candidate_stage(self, candidate_id: str, stage: str) -> CandidateRecord:
        with self._lock:
            raw = self._index["candidates"].get(candidate_id)
            if raw is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            raw["stage"] = stage
            if stage == "Shortlisted":
                raw["shortlisted"] = True
            self._save()
            return CandidateRecord.model_validate(self._with_defaults(raw))

    def update_candidate_shortlist(self, candidate_id: str, shortlisted: bool) -> CandidateRecord:
        with self._lock:
            raw = self._index["candidates"].get(candidate_id)
            if raw is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            raw["shortlisted"] = shortlisted
            if shortlisted and raw.get("stage") == "Applied":
                raw["stage"] = "Shortlisted"
            self._save()
            return CandidateRecord.model_validate(self._with_defaults(raw))

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        raw = self._index["candidates"].get(candidate_id)
        if raw is None:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return CandidateRecord.model_validate(self._with_defaults(raw))

    def list_candidates(self) -> list[CandidateRecord]:
        return [CandidateRecord.model_validate(self._with_defaults(item)) for item in self._state["candidates"]]

    def remove_candidate_if_orphaned(self, candidate_id: str) -> None:
        with self._lock:
            raw = self._index["candidates"].get(candidate_id)
            if raw is None or raw.get("document_ids"):
                return
            self._state["candidates"] = [row for row in self._state["candidates"] if row["id"] != candidate_id]
            self._state["notes"] = [note for note in self._state["notes"] if note.get("candidate_id") != candidate_id]
            self._index["candidates"].pop(candidate_id, None)
            self._save()

    @staticmethod
    def _with_defaults(raw: dict) -> dict:
        raw.setdefault("stage", "Applied")
        raw.setdefault("shortlisted", False)
        raw.setdefault("skills", [])
        raw.setdefault("pii_types_redacted", [])
        raw.setdefault("total_years_experience", 0.0)
        return raw

    # -- documents ---------------------------------------------------------

    def register_document(
        self, *, filename: str, entity_type: str, entity_id: str, entity_name: str, chunk_count: int
    ) -> DocumentRecord:
        document = DocumentRecord(
            id=self._new_id(),
            filename=filename,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            chunk_count=chunk_count,
            uploaded_at=self._now(),
        )
        with self._lock:
            raw = document.model_dump(mode="json")
            self._state["documents"].append(raw)
            self._index["documents"][document.id] = raw
            self._save()
        return document

    def update_document_chunk_count(self, document_id: str, chunk_count: int) -> DocumentRecord:
        with self._lock:
            raw = self._index["documents"].get(document_id)
            if raw is None:
                raise KeyError(f"Unknown document_id: {document_id}")
            raw["chunk_count"] = chunk_count
            self._save()
            return DocumentRecord.model_validate(raw)

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            before = len(self._state["documents"])
            self._state["documents"] = [row for row in self._state["documents"] if row["id"] != document_id]
            changed = len(self._state["documents"]) != before
            self._index["documents"].pop(document_id, None)
            for key in ("roles", "candidates"):
                for raw in self._state[key]:
                    doc_ids = [doc for doc in raw.get("document_ids", []) if doc != document_id]
                    if doc_ids != raw.get("document_ids", []):
                        raw["document_ids"] = doc_ids
                        changed = True
            if changed:
                self._save()

    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord.model_validate(item) for item in self._state["documents"]]

    # -- notes -------------------------------------------------------------

    def add_note(self, candidate_id: str, text: str) -> RecruiterNote:
        self.get_candidate(candidate_id)
        note = RecruiterNote(
            id=self._new_id(),
            candidate_id=candidate_id,
            text=text.strip(),
            created_at=self._now(),
        )
        with self._lock:
            self._state["notes"].append(note.model_dump(mode="json"))
            self._save()
        return note

    def list_notes(self, candidate_id: str) -> list[RecruiterNote]:
        notes = [
            RecruiterNote.model_validate(raw)
            for raw in self._state.get("notes", [])
            if raw.get("candidate_id") == candidate_id
        ]
        notes.sort(key=lambda note: note.created_at, reverse=True)
        return notes

    def count_notes(self, candidate_id: str) -> int:
        return sum(1 for raw in self._state.get("notes", []) if raw.get("candidate_id") == candidate_id)
