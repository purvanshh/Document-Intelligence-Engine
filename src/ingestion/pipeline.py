"""End-to-end ingestion and OCR pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from document_intelligence_engine.core.logging import get_logger
from ingestion.exceptions import EmptyOCROutputError
from ingestion.file_validator import validate_file
from ingestion.pdf_loader import load_document_images
from ocr.bbox_alignment import align_tokens_with_boxes
from ocr.ocr_engine import get_ocr_engine
from preprocessing.image_preprocessing import ImagePreprocessor


logger = get_logger(__name__)


def process_document(file_path: str) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    resolved_path = Path(file_path).expanduser().resolve()
    logger.info("file_received", extra={"file_path": str(resolved_path)})

    validation_start = time.perf_counter()
    validated_path = validate_file(resolved_path)
    logger.info(
        "file_validated",
        extra={
            "file_path": str(validated_path),
            "validation_time_ms": _elapsed_ms(validation_start),
        },
    )

    load_start = time.perf_counter()
    images = load_document_images(validated_path)
    logger.info(
        "document_loaded",
        extra={
            "file_path": str(validated_path),
            "page_count": len(images),
            "load_time_ms": _elapsed_ms(load_start),
        },
    )

    preprocessor = ImagePreprocessor()
    ocr_engine = get_ocr_engine()
    structured_output: list[dict[str, Any]] = []

    for page_number, image in enumerate(images, start=1):
        page_start = time.perf_counter()
        processed_image = preprocessor.preprocess(image)
        raw_tokens = ocr_engine.extract_tokens(processed_image)
        aligned_tokens = align_tokens_with_boxes(raw_tokens, processed_image.size)

        logger.info(
            "page_processed",
            extra={
                "file_path": str(validated_path),
                "page_number": page_number,
                "token_count": len(aligned_tokens),
                "page_time_ms": _elapsed_ms(page_start),
            },
        )

        for token in aligned_tokens:
            structured_output.append(
                {
                    "text": token["text"],
                    "bbox": token["bbox"],
                    "confidence": token["confidence"],
                }
            )

        if processed_image is not image:
            processed_image.close()
        image.close()

    if not structured_output:
        logger.error("empty_ocr_output", extra={"file_path": str(validated_path)})
        raise EmptyOCROutputError(f"OCR returned no tokens for document: {validated_path}")

    logger.info(
        "ocr_completed",
        extra={
            "file_path": str(validated_path),
            "total_tokens": len(structured_output),
            "total_time_ms": _elapsed_ms(started_at),
        },
    )
    return structured_output


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
