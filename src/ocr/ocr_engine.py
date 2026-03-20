"""
OCR Engine Module
-----------------
Wraps PaddleOCR (primary) with a Tesseract fallback.
Returns normalized tokens and bounding boxes per page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRToken:
    text: str
    bbox: List[int]          # [x_min, y_min, x_max, y_max] normalised 0-1000
    confidence: float = 1.0


class OCREngine:
    """
    Unified OCR wrapper.

    Usage:
        engine = OCREngine(backend="paddle")
        tokens = engine.run(image_array)
    """

    def __init__(self, backend: str = "paddle", lang: str = "en"):
        self.backend = backend.lower()
        self.lang = lang
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.backend == "paddle":
            try:
                from paddleocr import PaddleOCR  # type: ignore
                return PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            except ImportError:
                logger.warning("PaddleOCR not installed. Falling back to Tesseract.")
                self.backend = "tesseract"

        if self.backend == "tesseract":
            try:
                import pytesseract  # type: ignore  # noqa: F401
                return None  # pytesseract is function-based
            except ImportError as exc:
                raise ImportError("Neither PaddleOCR nor pytesseract is installed.") from exc

        raise ValueError(f"Unknown OCR backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> List[OCRToken]:
        """
        Run OCR on a single page image.

        Args:
            image: HxWx3 numpy array (RGB).

        Returns:
            List of OCRToken objects sorted by reading order (top-to-bottom, left-to-right).
        """
        if self.backend == "paddle":
            return self._run_paddle(image)
        return self._run_tesseract(image)

    def _run_paddle(self, image: np.ndarray) -> List[OCRToken]:
        import cv2  # type: ignore
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self._model.ocr(bgr, cls=True)

        h, w = image.shape[:2]
        tokens: List[OCRToken] = []
        for line in (result[0] or []):
            pts, (text, conf) = line
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [
                int(min(xs) / w * 1000),
                int(min(ys) / h * 1000),
                int(max(xs) / w * 1000),
                int(max(ys) / h * 1000),
            ]
            tokens.append(OCRToken(text=text, bbox=bbox, confidence=float(conf)))

        return sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))

    def _run_tesseract(self, image: np.ndarray) -> List[OCRToken]:
        import pytesseract  # type: ignore
        from PIL import Image

        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

        h, w = image.shape[:2]
        tokens: List[OCRToken] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = max(float(data["conf"][i]) / 100, 0.0)
            left = data["left"][i]
            top = data["top"][i]
            right = left + data["width"][i]
            bottom = top + data["height"][i]
            bbox = [
                int(left / w * 1000),
                int(top / h * 1000),
                int(right / w * 1000),
                int(bottom / h * 1000),
            ]
            tokens.append(OCRToken(text=text, bbox=bbox, confidence=conf))

        return tokens
