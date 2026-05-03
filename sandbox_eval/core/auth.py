from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header


@dataclass
class Actor:
    user_id: str
    role: str = "user"

    def can_write(self) -> bool:
        return "write" in (self.role or "") or self.is_admin()

    def is_admin(self) -> bool:
        return "admin" in (self.role or "")


def get_actor(
    x_user: Optional[str] = Header(default=None, alias="X-User"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
) -> Actor:
    user_id = x_user or "guest"
    role = x_role or "user"
    _ = x_api_key
    return Actor(user_id=user_id, role=role)

