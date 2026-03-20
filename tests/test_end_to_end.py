from __future__ import annotations

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def test_end_to_end_pipeline(monkeypatch):
    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)

    monkeypatch.setattr(
        "document_intelligence_engine.services.document_parser.process_document_with_metadata",
        lambda file_path, debug=False: {
            "file_path": file_path,
            "page_count": 1,
            "ocr_tokens": [
                {"text": "Invoice", "confidence": 0.98},
                {"text": "Number", "confidence": 0.97},
                {"text": "INV-1023", "confidence": 0.96},
                {"text": "Date", "confidence": 0.95},
                {"text": "2025-01-12", "confidence": 0.94},
                {"text": "Total", "confidence": 0.93},
                {"text": "Amount", "confidence": 0.92},
                {"text": "1200.50", "confidence": 0.91},
            ],
            "page_stats": [{"page_number": 1, "aligned_token_count": 8, "raw_token_count": 8}],
            "timing": {
                "validation": 1.0,
                "load": 2.0,
                "preprocessing": 3.0,
                "ocr": 4.0,
                "bbox_alignment": 1.5,
                "total": 11.5,
            },
            "debug": {"pages": []} if debug else {},
        },
    )

    result = parser_service.parse_file("invoice.pdf", debug=True)
    assert result["document"]["invoice_number"]["value"] == "INV-1023"
    assert result["metadata"]["timing"]["model"] >= 0.0
    assert "debug" in result
