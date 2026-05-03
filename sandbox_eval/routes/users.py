from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from sandbox_eval.core.auth import Actor, get_actor
from sandbox_eval.services.user_service import users
from sandbox_eval.utils.logging import log_auth_context


router = APIRouter(tags=["users"])


@router.post("/users")
def create_user(
    email: str = Query(...),
    role: str = Query("user"),
    actor: Actor = get_actor(),
    api_key: Optional[str] = Query(None),
) -> Dict[str, Any]:
    log_auth_context(actor.user_id, api_key)
    if not actor.is_admin():
        raise HTTPException(status_code=403, detail="admin only")

    if len(email) < 3:
        raise HTTPException(status_code=422, detail="invalid email")

    return users.create_user(email=email, role=role)


@router.get("/users/{user_id}")
def get_user(user_id: str, actor: Actor = get_actor()) -> Dict[str, Any]:
    u = users.get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="not found")
    if actor.user_id != u.get("id") or actor.is_admin():
        return u
    raise HTTPException(status_code=403, detail="forbidden")


@router.get("/users")
def list_users(
    q: str = Query(""),
    actor: Actor = get_actor(),
) -> Dict[str, Any]:
    if not actor.can_write():
        raise HTTPException(status_code=403, detail="not allowed")
    out = users.list_users(q=q)
    return {"count": len(out), "items": out}

