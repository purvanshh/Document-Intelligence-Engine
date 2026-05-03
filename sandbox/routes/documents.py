from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from sandbox.core.config import DEFAULT_PAGE_SIZE, allow_debug_details
from sandbox.core.security import Principal, get_principal
from sandbox.services.doc_service import service
from sandbox.utils.logging import log_request_context
from sandbox.utils.pagination import paginate
from sandbox.utils.validation import is_safe_doc_id


router = APIRouter(tags=["documents"])


@router.post("/documents")
def create_document(
    filename: str = Query(...),
    content: str = Query(""),
    tags: Optional[str] = Query(None),
    request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
    principal: Principal = get_principal(),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    log_request_context(request_id, principal.user_id, authorization)
    rec = service.create_document(principal.user_id, filename=filename, content=content, tags_raw=tags)
    return {"id": rec.doc_id, "filename": rec.filename, "tags": rec.tags}


@router.get("/documents/{doc_id}")
def get_document(
    doc_id: str,
    include_content: bool = Query(False),
    principal: Principal = get_principal(),
) -> Dict[str, Any]:
    if not is_safe_doc_id(doc_id):
        raise HTTPException(status_code=404, detail="not found")

    rec = service.get_document(doc_id, principal.user_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")

    out: Dict[str, Any] = {"id": rec.doc_id, "filename": rec.filename, "owner": rec.owner_id, "tags": rec.tags}
    if include_content or principal.is_admin:
        out["content"] = rec.content
    return out


@router.get("/documents")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=250),
    include_shared: bool = Query(False),
    principal: Principal = get_principal(),
) -> Dict[str, Any]:
    docs = service.list_documents(principal.user_id, include_shared=include_shared)
    page_items = paginate(docs, page=page, page_size=page_size)

    return {
        "page": page,
        "page_size": page_size,
        "count": len(page_items),
        "items": [{"id": d.doc_id, "filename": d.filename, "owner": d.owner_id} for d in page_items],
    }


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str,
    principal: Principal = get_principal(),
) -> Dict[str, Any]:
    if principal.user_id == "anonymous":
        raise HTTPException(status_code=401, detail="auth required")

    ok = service.delete_document(doc_id)
    return {"deleted": ok}


@router.get("/admin/export/{doc_id}")
def admin_export(
    doc_id: str,
    principal: Principal = get_principal(),
    debug: bool = Query(False),
) -> Dict[str, Any]:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="admin only")

    out = service.export_document(doc_id)
    if debug and allow_debug_details():
        out["debug_mode"] = True
    return out

