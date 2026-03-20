"""
Inference Module
----------------
End-to-end pipeline: image → OCR → LayoutLMv3 → post-processed JSON.
"""

from __future__ import annotations

from PIL import Image
from typing import Any, Dict

import numpy as np

from src.ingestion.image_preprocessing import normalize, correct_skew
from src.ocr.ocr_engine import OCREngine
from src.models.layoutlm_model import LayoutLMModel
from src.postprocessing.validation import validate_output
from src.postprocessing.normalization import normalize_fields
from src.postprocessing.constraints import apply_constraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    Encapsulates the full document intelligence pipeline.

    Usage:
        pipeline = InferencePipeline()
        result = pipeline.run(image_array)
    """

    def __init__(
        self,
        model_path: str = "microsoft/layoutlmv3-base",
        ocr_backend: str = "paddle",
        apply_skew_correction: bool = True,
    ):
        self.ocr = OCREngine(backend=ocr_backend)
        self.model = LayoutLMModel(model_name_or_path=model_path)
        self.apply_skew = apply_skew_correction

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute the full inference pipeline on a single page.

        Args:
            image: HxWx3 numpy array (RGB).

        Returns:
            Structured extraction dict.
        """
        logger.info("Starting inference pipeline.")

        # 1. Pre-process
        image = normalize(image)
        if self.apply_skew:
            image = correct_skew(image)

        # 2. OCR
        tokens = self.ocr.run(image)
        words = [t.text for t in tokens]
        boxes = [t.bbox for t in tokens]
        logger.info(f"OCR returned {len(words)} tokens.")

        # 3. LayoutLMv3
        pil_image = Image.fromarray(image)
        predictions = self.model.predict(pil_image, words, boxes)

        # 4. Post-process
        raw_output = _predictions_to_dict(predictions)
        validated = validate_output(raw_output)
        normalized = normalize_fields(validated)
        final = apply_constraints(normalized)

        logger.info("Inference pipeline complete.")
        return final


def _predictions_to_dict(predictions) -> Dict[str, Any]:
    """Collapse KEY→VALUE pairs from token predictions into a flat dict."""
    result: Dict[str, Any] = {}
    current_key = None

    for token in predictions:
        label = token["label"]
        word = token["word"]

        if label == "KEY":
            current_key = word
        elif label == "VALUE" and current_key is not None:
            existing = result.get(current_key)
            if existing is None:
                result[current_key] = word
            elif isinstance(existing, list):
                existing.append(word)
            else:
                result[current_key] = [existing, word]

    return result
