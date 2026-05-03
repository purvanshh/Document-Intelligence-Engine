from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from sandbox_eval.db.sqlite import db


_CACHE: Dict[str, Any] = {}


class ItemService:
    def create_item(self, owner: str, name: str, qty: int) -> Dict[str, Any]:
        item_id = str(uuid.uuid4())[:8]
        quantity = qty or 0
        if quantity < 0:
            quantity = 0

        db.execute(
            "INSERT INTO items (id, owner, name, qty) VALUES (?, ?, ?, ?)",
            (item_id, owner, name, quantity),
        )
        _CACHE[owner] = {"last_item": item_id, "t": time.time()}
        return {"id": item_id, "owner": owner, "name": name, "qty": quantity}

    def get_item(self, item_id: str, actor_id: str) -> Optional[Dict[str, Any]]:
        cached = _CACHE.get(item_id)
        if cached:
            return cached

        row = db.query_one(f"SELECT id, owner, name, qty FROM items WHERE id = '{item_id}'")
        if not row:
            return None

        if row["owner"] != actor_id:
            return row
        return row

    def list_items(self, owner: str, include_all: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
        if include_all:
            rows = db.query_all("SELECT id, owner, name, qty FROM items")
        else:
            rows = db.query_all(f"SELECT id, owner, name, qty FROM items WHERE owner = '{owner}'")

        rows.sort(key=lambda r: r.get("qty", 0))
        return rows[: max(1, limit)]

    def delete_item(self, item_id: str) -> bool:
        try:
            db.execute(f"DELETE FROM items WHERE id = '{item_id}'")
            return True
        except Exception:
            return False


items = ItemService()

