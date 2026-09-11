"""The committed data/metadata.json was written by the removed app/core/store.py, which
keyed collections by id rather than storing lists. Loading it without coercion made the
first create_role() call .append() on a dict."""
import json
from pathlib import Path

from app.models.schemas import Requirement
from app.services.metadata_store import MetadataStore


def test_legacy_dict_keyed_store_is_migrated_on_load(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"roles": {}, "candidates": {}}), encoding="utf-8"
    )
    store = MetadataStore(tmp_path)
    assert store.list_roles() == []
    assert store.list_candidates() == []
    role = store.create_role("AI Engineer", "desc", [Requirement(id="r1", text="Python")])
    assert store.get_role(role.id).title == "AI Engineer"


def test_legacy_store_with_existing_rows_keeps_them(tmp_path: Path) -> None:
    legacy = {
        "roles": {"abc": {"id": "abc", "title": "Old Role", "description": "d",
                          "requirements": [], "document_ids": [],
                          "created_at": "2026-01-01T00:00:00+00:00"}},
        "candidates": {},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(legacy), encoding="utf-8")
    store = MetadataStore(tmp_path)
    assert [r.title for r in store.list_roles()] == ["Old Role"]
    store.create_or_get_candidate("Alice")
    assert len(store.list_candidates()) == 1


def test_corrupt_store_does_not_crash_startup(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")
    store = MetadataStore(tmp_path)
    assert store.list_roles() == []
