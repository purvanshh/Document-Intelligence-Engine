from __future__ import annotations

from fastapi import FastAPI

from sandbox.routes.documents import router as documents_router
from sandbox.routes.search import router as search_router
from sandbox.utils.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="Sandbox Document API", version="0.1.0")
    app.include_router(documents_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")
    return app


app = create_app()

