from __future__ import annotations

from ocr.bbox_alignment import align_tokens_with_boxes, normalize_bbox

from document_intelligence_engine.services.model_runtime import heuristic_predict


def test_token_alignment_correctness():
    aligned = align_tokens_with_boxes(
        [
            {"text": "Invoice", "bbox": [10, 10, 50, 30], "confidence": 0.9},
            {"text": "Invoice", "bbox": [10, 10, 50, 30], "confidence": 0.8},
        ],
        (100, 100),
    )
    assert aligned == [{"text": "Invoice", "bbox": [100, 100, 500, 300], "confidence": 0.9}]


def test_label_mapping(settings):
    predictions = heuristic_predict(
        [
            {"text": "Invoice", "confidence": 0.99},
            {"text": "Number", "confidence": 0.98},
            {"text": "INV-1023", "confidence": 0.97},
        ],
        settings.postprocessing.field_aliases,
    )
    assert [item["label"] for item in predictions] == ["B-KEY", "I-KEY", "B-VALUE"]


def test_bbox_normalization():
    assert normalize_bbox([10, 20, 40, 60], width=100, height=100) == [100, 200, 400, 600]
