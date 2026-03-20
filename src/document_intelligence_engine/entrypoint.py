"""Project entrypoint."""

from __future__ import annotations

import uvicorn

from document_intelligence_engine.core.config import get_settings


def main() -> int:
    settings = get_settings()
    uvicorn.run(
        "document_intelligence_engine.api.app:create_app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        factory=True,
        workers=settings.api.workers,
    )
    return 0
