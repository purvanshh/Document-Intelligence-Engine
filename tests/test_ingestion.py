from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ingestion.exceptions import InvalidFileError
from ingestion.file_validator import validate_file
from ingestion.pipeline import process_document


class FakeOCREngine:
    def extract_tokens(self, image: Image.Image) -> list[dict[str, object]]:
        _ = image
        return [{"text": "Invoice", "bbox": [10, 20, 110, 60], "confidence": 0.98}]


def test_valid_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("ingestion.file_validator.pdfinfo_from_path", lambda _: {"Pages": 1})
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: [Image.new("RGB", (200, 100), color="white")],
    )
    monkeypatch.setattr("ingestion.pipeline.get_ocr_engine", lambda: FakeOCREngine())

    output = process_document(str(pdf_path))

    assert output
    assert output[0]["text"] == "Invoice"
    assert output[0]["bbox"] == [50, 200, 550, 600]


def test_corrupted_file(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"not-a-real-image")

    with pytest.raises(InvalidFileError):
        validate_file(corrupted_path)
