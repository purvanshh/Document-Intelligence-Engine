from __future__ import annotations

import pytest

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService
from ingestion.exceptions import EmptyOCROutputError


def test_empty_document_raises(monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr(
        "document_intelligence_engine.services.document_parser.process_document_with_metadata",
        lambda file_path, debug=False: (_ for _ in ()).throw(EmptyOCROutputError("empty document")),
    )

    with pytest.raises(EmptyOCROutputError):
        parser_service.parse_file("empty.pdf")


def test_noisy_ocr_generates_warning(monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr(
        "document_intelligence_engine.services.document_parser.process_document_with_metadata",
        lambda file_path, debug=False: {
            "file_path": file_path,
            "page_count": 1,
            "ocr_tokens": [{"text": "Inv0ice", "confidence": 0.3}],
            "page_stats": [{"page_number": 1, "aligned_token_count": 1, "raw_token_count": 1}],
            "timing": {
                "validation": 1.0,
                "load": 1.0,
                "preprocessing": 1.0,
                "ocr": 1.0,
                "bbox_alignment": 1.0,
                "total": 5.0,
            },
            "debug": {},
        },
    )

    result = parser_service.parse_file("noisy.pdf")
    assert "noisy_ocr_output" in result["metadata"]["warnings"]


def test_missing_key_fields_warning(monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr(
        "document_intelligence_engine.services.document_parser.process_document_with_metadata",
        lambda file_path, debug=False: {
            "file_path": file_path,
            "page_count": 2,
            "ocr_tokens": [
                {"text": "Date", "confidence": 0.99},
                {"text": "2025-01-12", "confidence": 0.98},
            ],
            "page_stats": [
                {"page_number": 1, "aligned_token_count": 2, "raw_token_count": 2},
                {"page_number": 2, "aligned_token_count": 0, "raw_token_count": 0},
            ],
            "timing": {
                "validation": 1.0,
                "load": 1.0,
                "preprocessing": 1.0,
                "ocr": 1.0,
                "bbox_alignment": 1.0,
                "total": 5.0,
            },
            "debug": {},
        },
    )

    result = parser_service.parse_file("missing-fields.pdf")
    assert "missing_required_fields" in result["metadata"]["warnings"]
