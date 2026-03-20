"""PDF and image loading utilities."""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from document_intelligence_engine.core.config import get_settings
from ingestion.exceptions import PDFLoadingError


def load_document_images(file_path: str | Path) -> list[Image.Image]:
    path = Path(file_path).expanduser().resolve()
    extension = path.suffix.lower()

    if extension == ".pdf":
        return _load_pdf_images(path)

    return [_load_image(path)]


def _load_pdf_images(file_path: Path) -> list[Image.Image]:
    settings = get_settings()
    try:
        images = convert_from_path(
            str(file_path),
            dpi=settings.ingestion.pdf_dpi,
            fmt="png",
            thread_count=1,
        )
    except Exception as exc:
        raise PDFLoadingError(f"Unable to convert PDF to images: {file_path}") from exc

    if not images:
        raise PDFLoadingError(f"PDF produced no pages: {file_path}")

    return [image.convert("RGB") for image in images]


def _load_image(file_path: Path) -> Image.Image:
    with Image.open(file_path) as image:
        return image.convert("RGB")
