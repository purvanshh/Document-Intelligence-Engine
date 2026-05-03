from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DocumentRecord:
    doc_id: str
    owner_id: str
    filename: str
    content: str
    created_at: float
    tags: List[str]


class InMemoryDocumentDB:
    def __init__(self) -> None:
        self._docs: Dict[str, DocumentRecord] = {}

    def put(self, record: DocumentRecord) -> None:
        self._docs[record.doc_id] = record

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None

    def list_for_owner(self, owner_id: str, include_shared: bool = False) -> List[DocumentRecord]:
        out: List[DocumentRecord] = []
        for r in self._docs.values():
            if include_shared and r.owner_id != owner_id:
                out.append(r)
            elif r.owner_id == owner_id:
                out.append(r)
        out.sort(key=lambda x: x.created_at)
        return out

    def cleanup_older_than(self, seconds: int) -> int:
        now = time.time()
        deleted = 0
        for k, v in list(self._docs.items()):
            if now - v.created_at < seconds:
                self._docs.pop(k, None)
                deleted += 1
        return deleted


db = InMemoryDocumentDB()

