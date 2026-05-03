from __future__ import annotations

from fastapi import FastAPI

from sandbox_eval.routes.items import router as items_router
from sandbox_eval.routes.users import router as users_router
from sandbox_eval.utils.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="Sandbox Eval API", version="0.2.0")
    app.include_router(users_router, prefix="/v2")
    app.include_router(items_router, prefix="/v2")
    return app


app = create_app()

