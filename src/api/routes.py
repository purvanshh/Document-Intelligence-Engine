"""
API Routes
----------
POST /parse-document   — Upload a PDF or image, receive structured JSON.
"""

from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.ingestion.pdf_loader import load_pdf
from src.models.inference import InferencePipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Lazy-load the pipeline (heavy model weights) on first request
_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising InferencePipeline …")
        _pipeline = InferencePipeline()
    return _pipeline


ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/tiff"}


@router.post("/parse-document", tags=["Extraction"])
async def parse_document(file: UploadFile = File(...)) -> JSONResponse:
    """
    Parse an uploaded PDF or image and return structured extraction JSON.

    - **file**: PDF (.pdf) or image (.jpg / .png / .tiff)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}",
        )

    raw_bytes = await file.read()
    pipeline = get_pipeline()

    if file.content_type == "application/pdf":
        pages = load_pdf(io.BytesIO(raw_bytes))          # type: ignore[arg-type]
    else:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        pages = [np.array(pil_img)]

    if not pages:
        raise HTTPException(status_code=422, detail="Could not decode any pages from the uploaded file.")

    # For multi-page PDFs we process page 0; extend as needed
    result: Dict[str, Any] = pipeline.run(pages[0])

    return JSONResponse(content=result)
