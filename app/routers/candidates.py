from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.models.schemas import (
    CandidateListResponse,
    CandidateUploadFailure,
    CandidateUploadResponse,
)
from app.routers.deps import get_services
from app.services.container import ServiceContainer
from app.services.document_parser import DocumentParseError, SUPPORTED_EXTENSIONS, extract_text

router = APIRouter(tags=["candidates"])
logger = logging.getLogger(__name__)


@router.post("/candidates/upload", response_model=CandidateUploadResponse)
async def upload_candidates(
    resumes: list[UploadFile] = File(...),
    services: ServiceContainer = Depends(get_services),
) -> CandidateUploadResponse:
    uploaded = []
    failed: list[CandidateUploadFailure] = []
    max_bytes = services.settings.max_upload_mb * 1024 * 1024

    for upload in resumes:
        filename = upload.filename or "resume.pdf"
        ext = Path(filename).suffix.lower()
        try:
            if ext not in SUPPORTED_EXTENSIONS:
                raise DocumentParseError(
                    f"Unsupported file type: {ext or 'unknown'}. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            raw = await upload.read()
            if not raw:
                raise DocumentParseError("Uploaded file is empty.")
            if len(raw) > max_bytes:
                raise DocumentParseError(f"{filename} exceeds size limit of {services.settings.max_upload_mb} MB.")

            text = extract_text(raw, filename)
            candidate_name = services.naming.derive_candidate_name(filename, text)
            item = services.candidates.ingest_candidate_resume(candidate_name, filename, text)
            uploaded.append(item)
        except (DocumentParseError, ValueError) as exc:
            logger.warning("Candidate upload skipped for %s: %s", filename, exc)
            failed.append(CandidateUploadFailure(filename=filename, error=str(exc)))
        except Exception as exc:
            logger.exception("Unexpected failure while processing candidate file: %s", filename)
            failed.append(
                CandidateUploadFailure(
                    filename=filename,
                    error=f"Unexpected ingestion failure: {exc}",
                )
            )

    if not uploaded:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No candidates were uploaded successfully.",
                "failed": [item.model_dump() for item in failed],
            },
        )

    return CandidateUploadResponse(candidates=uploaded, failed=failed)


@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(services: ServiceContainer = Depends(get_services)) -> CandidateListResponse:
    return CandidateListResponse(candidates=services.metadata.list_candidates())
