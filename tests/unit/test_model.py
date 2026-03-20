from __future__ import annotations

from document_intelligence_engine.domain.contracts import BoundingBox, OCRResult, OCRToken
from document_intelligence_engine.multimodal.layoutlmv3 import LayoutLMv3InferenceService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def test_forward_pass_shape_correctness(settings):
    service = LayoutLMv3InferenceService()
    result = service.predict(
        OCRResult(
            tokens=[
                OCRToken(
                    text="Invoice",
                    bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
                    confidence=0.9,
                    page_number=1,
                )
            ],
            engine="mock",
            language="en",
        )
    )
    assert len(result.labels) == 1
    assert len(result.confidences) == 1


def test_inference_output_format(settings):
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    output = model_service.predict(
        [
            {"text": "Invoice", "confidence": 0.99},
            {"text": "Number", "confidence": 0.98},
            {"text": "INV-1023", "confidence": 0.97},
        ]
    )
    assert all(set(item.keys()) == {"text", "label", "confidence"} for item in output)
