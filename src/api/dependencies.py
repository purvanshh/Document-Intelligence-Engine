"""API dependencies and runtime orchestration."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile

from document_intelligence_engine import __version__
from document_intelligence_engine.core.config import AppSettings, get_settings
from document_intelligence_engine.core.logging import get_logger
from ingestion.exceptions import EmptyOCROutputError, InvalidFileError, OCRExecutionError
from ingestion.file_validator import validate_file
from ingestion.pdf_loader import load_document_images
from ingestion.pipeline import process_document
from ocr.ocr_engine import OCREngine, get_ocr_engine
from postprocessing.pipeline import postprocess_predictions


logger = get_logger(__name__)


class APIError(Exception):
    """Base API exception."""

    def __init__(self, message: str, status_code: int, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or []


class InvalidUploadError(APIError):
    """Invalid upload request."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


class PayloadTooLargeError(APIError):
    """Request exceeds configured size limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=413)


class ModelLoadError(APIError):
    """Model failed to load."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503)


class ProcessingError(APIError):
    """Pipeline execution failure."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message, status_code=500, details=details)


@dataclass(slots=True)
class StagedUpload:
    path: Path
    filename: str
    content_type: str
    size_bytes: int


@dataclass(slots=True)
class RuntimeState:
    settings: AppSettings
    model_service: "LayoutAwareModelService"
    model_loaded: bool
    ocr_loaded: bool
    startup_error: str | None = None


class LayoutAwareModelService:
    """Startup-loaded inference service."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._loaded = False
        self._name = settings.model.layoutlmv3_model_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        checkpoint_path = self._settings.model.checkpoint_path
        if self._settings.model.startup_validate_checkpoint and checkpoint_path:
            resolved = Path(checkpoint_path).expanduser().resolve()
            if not resolved.exists():
                raise ModelLoadError(f"Model checkpoint not found: {resolved}")
            if resolved.is_file() and resolved.stat().st_size == 0:
                raise ModelLoadError(f"Model checkpoint is empty: {resolved}")
        self._loaded = True

    def predict(self, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._loaded:
            raise ModelLoadError("Model runtime is not loaded.")
        return _heuristic_predict(ocr_tokens, self._settings.postprocessing.field_aliases)

    def predict_text_only(self, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.predict(ocr_tokens)

    def predict_without_postprocessing(self, ocr_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.predict(ocr_tokens)


def build_runtime() -> RuntimeState:
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_loaded = False
    ocr_loaded = False
    startup_error: str | None = None

    try:
        model_service.load()
        model_loaded = True
    except APIError as exc:
        startup_error = exc.message
        logger.error("model_startup_failure", extra={"error": exc.message})

    try:
        _ = get_ocr_engine()
        ocr_loaded = True
    except Exception as exc:
        message = str(exc)
        startup_error = message if startup_error is None else f"{startup_error}; {message}"
        logger.error("ocr_startup_failure", extra={"error": message})

    return RuntimeState(
        settings=settings,
        model_service=model_service,
        model_loaded=model_loaded,
        ocr_loaded=ocr_loaded,
        startup_error=startup_error,
    )


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime


def get_request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid.uuid4().hex))


async def stage_upload(upload_file: UploadFile, settings: AppSettings) -> StagedUpload:
    safe_name = sanitize_filename(upload_file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in settings.ingestion.supported_extensions:
        raise InvalidUploadError("Invalid file format.")

    if upload_file.content_type not in settings.security.allowed_content_types:
        raise InvalidUploadError("Unsupported content type.")

    payload = await upload_file.read()
    max_size_bytes = settings.api.max_upload_size_mb * 1024 * 1024
    if not payload:
        raise InvalidUploadError("Uploaded file is empty.")
    if len(payload) > max_size_bytes:
        raise PayloadTooLargeError("Uploaded file exceeds configured size limit.")

    temp_dir = Path(tempfile.mkdtemp(prefix="api-upload-", dir=settings.paths.upload_dir))
    staged_path = temp_dir / safe_name
    staged_path.write_bytes(payload)

    try:
        validate_file(staged_path)
    except InvalidFileError as exc:
        cleanup_staged_upload(staged_path)
        raise InvalidUploadError(str(exc)) from exc

    return StagedUpload(
        path=staged_path,
        filename=safe_name,
        content_type=upload_file.content_type or "application/octet-stream",
        size_bytes=len(payload),
    )


async def process_staged_upload(
    staged_upload: StagedUpload,
    runtime: RuntimeState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.perf_counter()
    document_tokens = await asyncio.to_thread(process_document, str(staged_upload.path))
    raw_predictions = await asyncio.to_thread(runtime.model_service.predict, document_tokens)
    structured_document = await asyncio.to_thread(postprocess_predictions, raw_predictions)
    page_count = await asyncio.to_thread(_count_pages, staged_upload.path)

    confidence_summary = build_confidence_summary(structured_document)
    metadata = {
        "filename": staged_upload.filename,
        "content_type": staged_upload.content_type,
        "size_bytes": staged_upload.size_bytes,
        "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "confidence_summary": confidence_summary,
        "page_count": page_count,
    }
    return structured_document, metadata


async def process_batch_uploads(
    uploads: list[UploadFile],
    runtime: RuntimeState,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    semaphore = asyncio.Semaphore(runtime.settings.api.batch_concurrency)

    async def _worker(upload: UploadFile) -> tuple[dict[str, Any], dict[str, Any]]:
        async with semaphore:
            staged = await stage_upload(upload, runtime.settings)
            try:
                return await process_staged_upload(staged, runtime)
            finally:
                cleanup_staged_upload(staged.path)

    return await asyncio.gather(*[_worker(upload) for upload in uploads])


def sanitize_filename(filename: str | None) -> str:
    name = Path(filename or "document").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return sanitized[:128] or "document"


def cleanup_staged_upload(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path.parent, ignore_errors=True)
    except OSError:
        logger.warning("staged_upload_cleanup_failed", extra={"path": str(path)})


def build_confidence_summary(document: dict[str, Any]) -> dict[str, Any]:
    confidences = [
        float(payload.get("confidence", 0.0))
        for field_name, payload in document.items()
        if not str(field_name).startswith("_") and isinstance(payload, dict) and "confidence" in payload
    ]
    dropped_fields = sum(
        1
        for error in document.get("_errors", [])
        if isinstance(error, dict) and error.get("code") in {"low_confidence", "ablation_dropped_low_confidence"}
    )
    if not confidences:
        return {"average": 0.0, "minimum": 0.0, "maximum": 0.0, "kept_fields": 0, "dropped_fields": dropped_fields}
    return {
        "average": round(sum(confidences) / len(confidences), 6),
        "minimum": round(min(confidences), 6),
        "maximum": round(max(confidences), 6),
        "kept_fields": len(confidences),
        "dropped_fields": dropped_fields,
    }


def _count_pages(path: Path) -> int:
    try:
        return len(load_document_images(path))
    except Exception:
        return 0


def _heuristic_predict(
    ocr_tokens: list[dict[str, Any]],
    field_aliases: dict[str, str],
) -> list[dict[str, Any]]:
    if not ocr_tokens:
        return []

    normalized_tokens = [_normalize_token(token.get("text", "")) for token in ocr_tokens]
    alias_sequences = {
        alias: [part for part in alias.split(" ") if part]
        for alias in sorted(field_aliases, key=lambda item: len(item.split()), reverse=True)
    }
    labels = ["O"] * len(ocr_tokens)
    spans: list[tuple[int, int]] = []

    index = 0
    while index < len(ocr_tokens):
        matched = False
        for alias, alias_parts in alias_sequences.items():
            end_index = index + len(alias_parts)
            if end_index > len(ocr_tokens):
                continue
            if normalized_tokens[index:end_index] == alias_parts:
                spans.append((index, end_index))
                labels[index] = "B-KEY"
                for offset in range(index + 1, end_index):
                    labels[offset] = "I-KEY"
                index = end_index
                matched = True
                break
        if not matched:
            index += 1

    for span_index, (_, span_end) in enumerate(spans):
        next_key_start = spans[span_index + 1][0] if span_index + 1 < len(spans) else len(ocr_tokens)
        value_start = span_end
        while value_start < next_key_start and _normalize_token(ocr_tokens[value_start].get("text", "")) in {"", ":"}:
            value_start += 1
        if value_start >= next_key_start:
            continue
        labels[value_start] = "B-VALUE"
        for value_index in range(value_start + 1, next_key_start):
            if _normalize_token(ocr_tokens[value_index].get("text", "")) == "":
                continue
            labels[value_index] = "I-VALUE"

    return [
        {
            "text": str(token.get("text", "")),
            "label": labels[index],
            "confidence": round(float(token.get("confidence", 0.0)), 6),
        }
        for index, token in enumerate(ocr_tokens)
    ]


def _normalize_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9#]+", "", str(text).lower().strip())
