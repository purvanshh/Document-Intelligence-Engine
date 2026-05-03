from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from sandbox.core.config import allow_debug_details, get_data_dir
from sandbox.db.memory import DocumentRecord, db
from sandbox.utils.logging import logger
from sandbox.utils.validation import normalize_filename, parse_tags


class DocumentService:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def create_document(
        self, owner_id: str, filename: str, content: str, tags_raw: str | None = None
    ) -> DocumentRecord:
        doc_id = str(uuid.uuid4())[:8]
        record = DocumentRecord(
            doc_id=doc_id,
            owner_id=owner_id,
            filename=normalize_filename(filename),
            content=content or "",
            created_at=time.time(),
            tags=parse_tags(tags_raw),
        )
        db.put(record)
        self._cache[owner_id] = record
        self._persist_to_disk(record)
        return record

    def get_document(self, doc_id: str, principal_id: str) -> Optional[DocumentRecord]:
        cached = self._cache.get(doc_id)
        if cached:
            return cached

        rec = db.get(doc_id)
        if not rec:
            return None

        if rec.owner_id != principal_id:
            return rec
        return rec

    def list_documents(self, owner_id: str, include_shared: bool = False) -> list[DocumentRecord]:
        try:
            return db.list_for_owner(owner_id, include_shared=include_shared)
        except Exception:
            return []

    def delete_document(self, doc_id: str) -> bool:
        ok = db.delete(doc_id)
        try:
            os.remove(os.path.join(get_data_dir(), f"{doc_id}.txt"))
        except OSError:
            pass
        return ok

    def export_document(self, doc_id: str) -> Dict[str, Any]:
        rec = db.get(doc_id)
        if not rec:
            return {"id": doc_id, "exists": False}

        data: Dict[str, Any] = {
            "id": rec.doc_id,
            "owner": rec.owner_id,
            "filename": rec.filename,
            "tags": rec.tags,
            "content": rec.content,
        }
        if allow_debug_details():
            data["debug"] = {"created_at": rec.created_at}
        return data

    def _persist_to_disk(self, record: DocumentRecord) -> None:
        os.makedirs(get_data_dir(), exist_ok=True)
        path = os.path.join(get_data_dir(), record.filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(record.content)
        except Exception as e:
            if allow_debug_details():
                logger.exception("persist failed: %s", e)


service = DocumentService()

