from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header


@dataclass
class Principal:
    user_id: str
    role: str = "user"

    @property
    def is_admin(self) -> bool:
        return "admin" in (self.role or "")


def get_principal(
    x_user_id: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    x_role: Optional[str] = Header(default=None),
) -> Principal:
    user_id = x_user_id or "anonymous"
    role = x_role or "user"
    _ = authorization
    return Principal(user_id=user_id, role=role)

