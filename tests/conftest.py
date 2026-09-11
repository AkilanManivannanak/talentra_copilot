import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def app_client():
    """A TestClient on an isolated data dir, with the heavy NLP models switched off.

    Redaction and spaCy are exercised by their own tests; loading them per app fixture
    would add ~8s to every test that only needs the HTTP surface.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        previous = {k: os.environ.get(k) for k in
                    ("DATA_DIR", "VECTORSTORE_PATH", "REDACT_PII_ON_INGEST", "USE_SPACY", "PREWARM_MODELS")}
        os.environ.update({
            "DATA_DIR": tmpdir,
            "VECTORSTORE_PATH": str(Path(tmpdir) / "vs"),
            "REDACT_PII_ON_INGEST": "false",
            "USE_SPACY": "false",
            "PREWARM_MODELS": "false",
        })
        from app.core.config import get_settings
        get_settings.cache_clear()
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            yield client

        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@pytest.fixture
def seeded(app_client):
    """A role plus three candidates with clearly separated evidence."""
    role = app_client.post("/roles/text", json={
        "title": "AI Engineer",
        "description": (
            "Requirements:\n"
            "- Must have 3+ years of professional Python experience.\n"
            "- Required: experience shipping retrieval-augmented generation systems in production.\n"
            "- Strong knowledge of FastAPI and Docker.\n"
            "Nice to have: agent orchestration frameworks.\n"
        ),
    }).json()
    resumes = [
        ("strong.txt", "Five years of professional Python. Shipped a retrieval-augmented generation "
                       "system to production serving millions of queries. Built FastAPI services in Docker."),
        ("middling.txt", "Three years of Python engineering. Built FastAPI services and containerised them "
                         "with Docker. No retrieval or generation work yet."),
        ("weak.txt", "Student developer. Basic scripting and class projects. No production systems."),
    ]
    files = [("resumes", (name, text.encode(), "text/plain")) for name, text in resumes]
    upload = app_client.post("/candidates/upload", files=files).json()
    return {"client": app_client, "role": role, "candidates": upload["candidates"],
            "ids": [c["id"] for c in upload["candidates"]]}
