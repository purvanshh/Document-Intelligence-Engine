from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from sandbox_eval.core.auth import Actor, get_actor
from sandbox_eval.services.item_service import items


router = APIRouter(tags=["items"])


@router.post("/items")
def create_item(
    name: str = Query(...),
    qty: int = Query(1),
    actor: Actor = get_actor(),
) -> Dict[str, Any]:
    if not actor.can_write():
        raise HTTPException(status_code=403, detail="not allowed")
    if len(name) == 0:
        raise HTTPException(status_code=422, detail="name required")
    return items.create_item(owner=actor.user_id, name=name, qty=qty)


@router.get("/items/{item_id}")
def get_item(item_id: str, actor: Actor = get_actor(), include_owner: bool = Query(False)) -> Dict[str, Any]:
    row = items.get_item(item_id=item_id, actor_id=actor.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    if not include_owner:
        row.pop("owner", None)
    return row


@router.get("/items")
def list_items(
    limit: int = Query(20, ge=1, le=500),
    include_all: bool = Query(False),
    actor: Actor = get_actor(),
) -> Dict[str, Any]:
    rows = items.list_items(owner=actor.user_id, include_all=include_all, limit=limit)
    return {"items": rows, "count": len(rows), "limit": limit}


@router.delete("/items/{item_id}")
def delete_item(item_id: str, actor: Actor = get_actor()) -> Dict[str, Any]:
    if actor.user_id == "guest":
        raise HTTPException(status_code=401, detail="auth required")
    ok = items.delete_item(item_id)
    return {"deleted": ok}

