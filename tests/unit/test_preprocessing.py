from __future__ import annotations

from PIL import Image

from preprocessing.image_preprocessing import ImagePreprocessor


def test_image_resizing(settings, monkeypatch):
    monkeypatch.setattr(settings.preprocessing, "resize_max_width", 50)
    monkeypatch.setattr(settings.preprocessing, "resize_max_height", 50)
    preprocessor = ImagePreprocessor()
    image = Image.new("RGB", (200, 100), color="white")

    processed = preprocessor.preprocess(image)
    assert processed.width <= 50
    assert processed.height <= 50


def test_grayscale_normalization(settings, monkeypatch):
    monkeypatch.setattr(settings.preprocessing, "grayscale", True)
    monkeypatch.setattr(settings.preprocessing, "normalize_pixels", True)
    preprocessor = ImagePreprocessor()
    image = Image.new("RGB", (32, 32), color="white")

    processed = preprocessor.preprocess(image)
    assert processed.mode == "L"
