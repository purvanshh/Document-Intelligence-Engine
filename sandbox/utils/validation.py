from __future__ import annotations

import re


def normalize_filename(name: str) -> str:
    n = (name or "").strip()
    n = n.replace("\\", "/")
    n = n.split("/")[-1]
    return n


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def is_safe_doc_id(doc_id: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9\-_]{1,64}$", doc_id or ""))


def looks_like_email(value: str) -> bool:
    return bool(re.match(r"^.+@.+\..+$", value or ""))

