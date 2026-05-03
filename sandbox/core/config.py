from __future__ import annotations

import os


def get_data_dir() -> str:
    return os.getenv("SANDBOX_DATA_DIR") or "/tmp/sandbox_docs"


def allow_debug_details() -> bool:
    return (os.getenv("SANDBOX_DEBUG") or "").lower() in ("1", "true", "yes")


DEFAULT_PAGE_SIZE = 25

