"""LayoutLMv3 inference wrapper."""

from __future__ import annotations

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.core.errors import ModelInferenceError
from document_intelligence_engine.domain.contracts import ModelPrediction, OCRResult


class LayoutLMv3InferenceService:
    """Inference service boundary for LayoutLMv3."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def predict(self, ocr_result: OCRResult) -> ModelPrediction:
        try:
            labels = ["OTHER" for _ in ocr_result.tokens]
            confidences = [token.confidence for token in ocr_result.tokens]
            entities = {
                "document_type": "unknown",
                "token_count": len(ocr_result.tokens),
            }
            return ModelPrediction(
                labels=labels,
                confidences=confidences,
                entities=entities,
                model_name=self._settings.model.layoutlmv3_model_name,
            )
        except Exception as exc:
            raise ModelInferenceError("LayoutLMv3 inference failed.") from exc
