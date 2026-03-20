from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ingestion.exceptions import EmptyOCROutputError
from ingestion.pipeline import process_document
from ocr.bbox_alignment import align_tokens_with_boxes, normalize_bbox
from ocr.ocr_engine import OCREngine


class EmptyBackend:
    def ocr(self, image_array, cls=True):  # noqa: ANN001, FBT002
        _ = image_array
        _ = cls
        return [[]]


class EmptyEngine:
    def extract_tokens(self, image: Image.Image) -> list[dict[str, object]]:
        _ = image
        return []


def test_empty_ocr_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    monkeypatch.setattr("ingestion.pipeline.get_ocr_engine", lambda: EmptyEngine())

    with pytest.raises(EmptyOCROutputError):
        process_document(str(image_path))


def test_ocr_engine_empty_backend() -> None:
    engine = OCREngine(backend=EmptyBackend())
    image = Image.new("RGB", (32, 32), color="white")
    assert engine.extract_tokens(image) == []


def test_normalize_bbox_and_alignment() -> None:
    assert normalize_bbox([10, 10, 50, 50], width=100, height=100) == [100, 100, 500, 500]
    aligned = align_tokens_with_boxes(
        [{"text": "A", "bbox": [10, 10, 50, 50], "confidence": 0.9}],
        image_size=(100, 100),
    )
    assert aligned == [{"text": "A", "bbox": [100, 100, 500, 500], "confidence": 0.9}]
