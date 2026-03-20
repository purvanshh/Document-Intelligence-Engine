from __future__ import annotations

from document_intelligence_engine.domain.contracts import BoundingBox, OCRResult, OCRToken
from document_intelligence_engine.multimodal.layoutlmv3 import LayoutLMv3InferenceService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def test_forward_pass_shape_correctness():
    service = LayoutLMv3InferenceService()
    ocr_result = OCRResult(
        tokens=[
            OCRToken(
                text="Invoice",
                bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
                confidence=0.9,
                page_number=1,
            ),
            OCRToken(
                text="INV-1023",
                bbox=BoundingBox(x0=10, y0=0, x1=20, y1=10),
                confidence=0.95,
                page_number=1,
            ),
        ],
        engine="mock",
        language="en",
    )

    result = service.predict(ocr_result)

    assert len(result.labels) == len(ocr_result.tokens)
    assert len(result.confidences) == len(ocr_result.tokens)
    assert result.entities["token_count"] == len(ocr_result.tokens)


def test_inference_output_format(settings):
    model_service = LayoutAwareModelService(settings)
    model_service.load()

    predictions = model_service.predict(
        [
            {"text": "Invoice", "confidence": 0.99},
            {"text": "Number", "confidence": 0.98},
            {"text": "INV-1023", "confidence": 0.97},
        ]
    )

    assert predictions == [
        {"text": "Invoice", "label": "B-KEY", "confidence": 0.99},
        {"text": "Number", "label": "I-KEY", "confidence": 0.98},
        {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.97},
    ]
