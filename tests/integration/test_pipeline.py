from __future__ import annotations

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def test_full_pipeline_valid_document(sample_image_path, mock_ocr_output, monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: type(
            "BatchOCR",
            (),
            {"extract_batch_tokens": lambda self, images: [mock_ocr_output[:]]},
        )(),
    )

    result = parser_service.parse_file(sample_image_path)
    assert result["document"]["invoice_number"]["value"] == "INV-1023"
    assert result["document"]["date"]["value"] == "2025-01-12"
    assert result["document"]["total_amount"]["value"] == 1200.5
    assert result["metadata"]["page_count"] == 1


def test_full_pipeline_noisy_ocr(sample_image_path, monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    noisy_tokens = [{"text": "Inv0ice", "bbox": [10, 10, 40, 30], "confidence": 0.3}]
    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: type(
            "BatchOCR",
            (),
            {"extract_batch_tokens": lambda self, images: [noisy_tokens[:]]},
        )(),
    )

    result = parser_service.parse_file(sample_image_path)
    assert "noisy_ocr_output" in result["metadata"]["warnings"]
    assert result["document"]["_errors"]


def test_full_pipeline_missing_fields(sample_image_path, monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    missing_tokens = [
        {"text": "Date", "bbox": [10, 10, 40, 30], "confidence": 0.99},
        {"text": "2025-01-12", "bbox": [50, 10, 120, 30], "confidence": 0.98},
    ]
    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: type(
            "BatchOCR",
            (),
            {"extract_batch_tokens": lambda self, images: [missing_tokens[:]]},
        )(),
    )

    result = parser_service.parse_file(sample_image_path)
    assert "missing_required_fields" in result["metadata"]["warnings"]
