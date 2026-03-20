from __future__ import annotations

from PIL import Image

from ocr.ocr_engine import OCREngine


class MockBackend:
    def ocr(self, image_array, cls=True):  # noqa: ANN001, FBT002
        _ = image_array
        _ = cls
        return [
            [
                (
                    [[10, 10], [80, 10], [80, 30], [10, 30]],
                    ("Invoice", 0.98),
                )
            ]
        ]


class EmptyBackend:
    def ocr(self, image_array, cls=True):  # noqa: ANN001, FBT002
        _ = image_array
        _ = cls
        return [[]]


def test_mock_ocr_output():
    engine = OCREngine(backend=MockBackend())
    tokens = engine.extract_tokens(Image.new("RGB", (100, 100), color="white"))
    assert tokens == [{"text": "Invoice", "bbox": [10, 10, 80, 30], "confidence": 0.98}]


def test_empty_ocr_case():
    engine = OCREngine(backend=EmptyBackend())
    tokens = engine.extract_tokens(Image.new("RGB", (100, 100), color="white"))
    assert tokens == []


def test_batch_ocr_case():
    engine = OCREngine(backend=MockBackend())
    images = [Image.new("RGB", (100, 100), color="white"), Image.new("RGB", (100, 100), color="white")]
    batch = engine.extract_batch_tokens(images)
    assert len(batch) == 2
    assert all(item[0]["text"] == "Invoice" for item in batch)
