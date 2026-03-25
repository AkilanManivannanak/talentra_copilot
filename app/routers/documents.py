from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.schemas import DocumentListResponse
from app.routers.deps import get_services
from app.services.container import ServiceContainer

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(services: ServiceContainer = Depends(get_services)) -> DocumentListResponse:
    return DocumentListResponse(documents=services.metadata.list_documents())
