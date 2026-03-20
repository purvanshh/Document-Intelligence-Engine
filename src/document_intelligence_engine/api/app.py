"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from document_intelligence_engine.api.routes.documents import router as document_router
from document_intelligence_engine.api.routes.health import router as health_router
from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.errors import (
    DocumentEngineError,
    InvalidInputError,
    ModelInferenceError,
    OCRProcessingError,
)
from document_intelligence_engine.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    logger = get_logger(__name__)

    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(document_router)

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(_: Request, exc: InvalidInputError) -> JSONResponse:
        logger.error("invalid_input", extra={"error": str(exc)})
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OCRProcessingError)
    async def ocr_handler(_: Request, exc: OCRProcessingError) -> JSONResponse:
        logger.error("ocr_failure", extra={"error": str(exc)})
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ModelInferenceError)
    async def inference_handler(_: Request, exc: ModelInferenceError) -> JSONResponse:
        logger.error("model_inference_failure", extra={"error": str(exc)})
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(DocumentEngineError)
    async def generic_handler(_: Request, exc: DocumentEngineError) -> JSONResponse:
        logger.error("application_error", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
