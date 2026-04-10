from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any


_lock = threading.Lock()


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"roles": {}, "candidates": {}}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class MetadataStore:
    def __init__(self, path: Path):
        self._path = path
        with _lock:
            self._data = _load(path)

    def _flush(self) -> None:
        _save(self._path, self._data)

    # ── Roles ────────────────────────────────────────────────────────────
    def create_role(self, title: str, description: str, requirements: list[str]) -> dict:
        role_id = str(uuid.uuid4())[:8]
        role = {"id": role_id, "title": title, "description": description, "requirements": requirements}
        with _lock:
            self._data["roles"][role_id] = role
            self._flush()
        return role

    def get_role(self, role_id: str) -> dict | None:
        return self._data["roles"].get(role_id)

    def list_roles(self) -> list[dict]:
        return list(self._data["roles"].values())

    # ── Candidates ───────────────────────────────────────────────────────
    def create_candidate(self, name: str, filename: str, text: str,
                         skills: list[str] | None = None,
                         preprocessed: dict | None = None) -> dict:
        cid = str(uuid.uuid4())[:8]
        candidate = {
            "id": cid,
            "name": name,
            "filename": filename,
            "text": text,
            "skills": skills or [],
            "preprocessed": preprocessed or {},
            "ats_stage": "new",
            "notes": [],
        }
        with _lock:
            self._data["candidates"][cid] = candidate
            self._flush()
        return candidate

    def get_candidate(self, cid: str) -> dict | None:
        return self._data["candidates"].get(cid)

    def list_candidates(self) -> list[dict]:
        return list(self._data["candidates"].values())

    def update_candidate(self, cid: str, **kwargs) -> dict | None:
        with _lock:
            c = self._data["candidates"].get(cid)
            if c is None:
                return None
            c.update(kwargs)
            self._flush()
        return c

    def delete_candidate(self, cid: str) -> bool:
        with _lock:
            if cid not in self._data["candidates"]:
                return False
            del self._data["candidates"][cid]
            self._flush()
        return True

    def add_note(self, cid: str, note: str) -> dict | None:
        with _lock:
            c = self._data["candidates"].get(cid)
            if c is None:
                return None
            c.setdefault("notes", []).append(note)
            self._flush()
        return c
