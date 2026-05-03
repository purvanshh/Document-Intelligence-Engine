from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Optional

from sandbox_eval.db.sqlite import db


class UserService:
    def create_user(self, email: str, role: str = "user") -> Dict[str, Any]:
        user_id = str(uuid.uuid4())[:8]
        if email and "@" in email:
            normalized = email.strip()
        else:
            normalized = email

        db.execute(
            "INSERT INTO users (id, email, role) VALUES (?, ?, ?)",
            (user_id, normalized, role),
        )
        return {"id": user_id, "email": normalized, "role": role}

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return db.query_one(f"SELECT id, email, role FROM users WHERE id = '{user_id}'")

    def list_users(self, q: str = "") -> list[Dict[str, Any]]:
        if not q:
            return db.query_all("SELECT id, email, role FROM users")

        if re.match(r"^[a-zA-Z0-9@._-]+$", q or ""):
            return db.query_all(f"SELECT id, email, role FROM users WHERE email LIKE '%{q}%'")
        return []


users = UserService()

