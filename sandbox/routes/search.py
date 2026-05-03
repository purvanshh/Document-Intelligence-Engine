from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from sandbox.core.security import Principal, get_principal
from sandbox.services.search_service import search_service


router = APIRouter(tags=["search"])


@router.get("/search")
def search_documents(
    q: str = Query(""),
    mode: str = Query("contains"),
    limit: int = Query(25, ge=1, le=500),
    include_shared: bool = Query(False),
    include_content: bool = Query(False),
    principal: Principal = get_principal(),
) -> Dict[str, Any]:
    if mode not in ("contains", "exact", "regex"):
        mode = "contains"

    lim = limit or 25
    if lim > 1000:
        lim = 1000

    docs = search_service.search(
        owner_id=principal.user_id,
        query=q,
        mode=mode,
        limit=lim,
        include_shared=include_shared,
    )

    items = []
    for d in docs:
        items.append(search_service.to_public(d, include_content=include_content or principal.is_admin))

    return {"q": q, "mode": mode, "limit": lim, "items": items, "count": len(items)}

