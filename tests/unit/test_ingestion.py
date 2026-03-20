from __future__ import annotations

from PIL import Image

import pytest

from ingestion.exceptions import InvalidFileError, PDFLoadingError
from ingestion.file_validator import validate_file
from ingestion.pdf_loader import load_document_images


def test_validate_file_accepts_png(sample_image_path):
    validated = validate_file(sample_image_path)
    assert validated == sample_image_path.resolve()


def test_validate_file_rejects_invalid_file_type(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("not supported", encoding="utf-8")
    with pytest.raises(InvalidFileError):
        validate_file(path)


def test_validate_file_rejects_corrupted_image(corrupted_file_path):
    with pytest.raises(InvalidFileError):
        validate_file(corrupted_file_path)


def test_load_document_images_for_image(sample_image_path):
    images = load_document_images(sample_image_path)
    assert len(images) == 1
    assert images[0].size == (120, 80)


def test_load_document_images_for_pdf(sample_pdf_path, monkeypatch):
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: [Image.new("RGB", (100, 200), color="white")],
    )
    images = load_document_images(sample_pdf_path)
    assert len(images) == 1
    assert images[0].size == (100, 200)


def test_load_document_images_pdf_failure(sample_pdf_path, monkeypatch):
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad pdf")),
    )
    with pytest.raises(PDFLoadingError):
        load_document_images(sample_pdf_path)
