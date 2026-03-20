"""
FastAPI Application – Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Document Intelligence Engine",
    description=(
        "Layout-aware multimodal document parsing system. "
        "Converts PDFs and images into structured JSON."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Document Intelligence Engine API started.")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "document-intelligence-engine"}
