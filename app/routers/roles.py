from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.models.schemas import EvaluateRoleRequest, EvaluateRoleResponse, RoleCreateRequest, RoleListResponse, RoleRecord
from app.routers.deps import get_services
from app.services.container import ServiceContainer
from app.services.document_parser import DocumentParseError, SUPPORTED_EXTENSIONS, extract_text

router = APIRouter(tags=["roles"])


@router.post("/roles/text", response_model=RoleRecord)
async def create_role_from_text(
    body: RoleCreateRequest,
    services: ServiceContainer = Depends(get_services),
) -> RoleRecord:
    return services.roles.create_role_from_text(body.title, body.description)


@router.post("/roles/upload", response_model=RoleRecord)
async def create_role_from_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    services: ServiceContainer = Depends(get_services),
) -> RoleRecord:
    raw = await file.read()
    filename = file.filename or "role.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{ext}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported JD file type: .{ext or 'unknown'}")
    try:
        text = extract_text(raw, filename)
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    resolved_title = title.strip() or services.naming.derive_role_title(filename, text)
    return services.roles.create_role_from_document(resolved_title, filename, text)


@router.get("/roles", response_model=RoleListResponse)
async def list_roles(services: ServiceContainer = Depends(get_services)) -> RoleListResponse:
    return RoleListResponse(roles=services.metadata.list_roles())


@router.post("/roles/{role_id}/evaluate", response_model=EvaluateRoleResponse)
async def evaluate_role(
    role_id: str,
    body: EvaluateRoleRequest,
    services: ServiceContainer = Depends(get_services),
) -> EvaluateRoleResponse:
    return services.ranking.evaluate_role(
        role_id=role_id,
        candidate_ids=body.candidate_ids,
        top_k_per_requirement=body.top_k_per_requirement,
    )
