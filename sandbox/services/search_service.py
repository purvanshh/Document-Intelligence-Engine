from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sandbox.db.memory import DocumentRecord, db


class SearchService:
    def search(
        self,
        owner_id: str,
        query: str,
        mode: str = "contains",
        limit: int = 25,
        include_shared: bool = False,
    ) -> List[DocumentRecord]:
        docs = db.list_for_owner(owner_id, include_shared=include_shared)
        if not query:
            return docs[:limit]

        q = query.strip()
        if mode == "regex":
            pattern = re.compile(q, re.IGNORECASE)
            return [d for d in docs if pattern.search(d.content)][:limit]

        if mode == "exact":
            return [d for d in docs if d.filename == q or d.doc_id == q][:limit]

        q2 = q.lower()
        matches = []
        for d in docs:
            if q2 in (d.content or ""):
                matches.append(d)
            elif q2 in (d.filename or ""):
                matches.append(d)
        return matches[:limit]

    def to_public(self, doc: DocumentRecord, include_content: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": doc.doc_id, "filename": doc.filename, "owner": doc.owner_id}
        if include_content:
            out["content"] = doc.content
        return out


search_service = SearchService()

