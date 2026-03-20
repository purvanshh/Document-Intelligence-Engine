from __future__ import annotations

from PIL import Image

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def test_multi_page_document_conflict_detection(sample_pdf_path, mock_ocr_output, monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr("ingestion.file_validator.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "ingestion.pdf_loader.convert_from_path",
        lambda *args, **kwargs: [
            Image.new("RGB", (100, 100), color="white"),
            Image.new("RGB", (100, 100), color="white"),
        ],
    )
    page_one = mock_ocr_output[:3]
    page_two = [
        {"text": "Invoice", "bbox": [10, 10, 80, 30], "confidence": 0.99},
        {"text": "Number", "bbox": [90, 10, 170, 30], "confidence": 0.98},
        {"text": "INV-2048", "bbox": [180, 10, 280, 30], "confidence": 0.97},
    ]
    monkeypatch.setattr(
        "ingestion.pipeline.get_ocr_engine",
        lambda: type(
            "BatchOCR",
            (),
            {"extract_batch_tokens": lambda self, images: [page_one[:], page_two[:]]},
        )(),
    )

    result = parser_service.parse_file(sample_pdf_path)
    assert result["metadata"]["page_count"] == 2
    assert "multi_page_inconsistency" in result["metadata"]["warnings"]


def test_pipeline_output_schema(sample_image_path, mock_ocr_output, monkeypatch):
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

    result = parser_service.parse_file(sample_image_path, debug=True)
    assert set(result.keys()) >= {"document", "metadata", "debug"}
    assert set(result["metadata"]["timing"].keys()) >= {
        "validation",
        "load",
        "preprocessing",
        "ocr",
        "bbox_alignment",
        "model",
        "postprocessing",
        "total",
    }
