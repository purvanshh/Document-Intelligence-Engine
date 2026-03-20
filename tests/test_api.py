from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class DummyModelService:
    loaded = True
    version = "test"
    device = "cpu"

    def predict(self, tokens):
        _ = tokens
        return [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.97},
            {"text": "Number", "label": "I-KEY", "confidence": 0.96},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.95},
        ]


def test_health_endpoint(monkeypatch):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}


def test_parse_document_valid_request(monkeypatch):
    app = create_app()

    with TestClient(app) as client:
        app.state.runtime.model_service = DummyModelService()
        monkeypatch.setattr(
            app.state.runtime.parser_service,
            "parse_file",
            lambda file_path, debug=False: {
                "document": {
                    "invoice_number": {"value": "INV-1023", "confidence": 0.95, "valid": True},
                    "_errors": [],
                    "_constraint_flags": [],
                },
                "metadata": {
                    "filename": "invoice.png",
                    "page_count": 1,
                    "ocr_token_count": 3,
                    "confidence_summary": {
                        "average": 0.95,
                        "minimum": 0.95,
                        "maximum": 0.95,
                        "kept_fields": 1,
                        "dropped_fields": 0,
                    },
                    "timing": {
                        "validation": 1.0,
                        "load": 1.0,
                        "preprocessing": 1.0,
                        "ocr": 1.0,
                        "bbox_alignment": 1.0,
                        "model": 1.0,
                        "postprocessing": 1.0,
                        "total": 7.0,
                    },
                    "warnings": [],
                    "model": {"name": "dummy", "version": "test", "device": "cpu"},
                },
            },
        )
        response = client.post(
            "/parse-document",
            files={"file": ("invoice.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["document"]["invoice_number"]["value"] == "INV-1023"
        assert payload["metadata"]["page_count"] == 1
        assert payload["metadata"]["timing"]["total"] == 7.0


def test_parse_document_invalid_file():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/parse-document",
            files={"file": ("bad.txt", b"invalid", "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid file format"
        assert response.json()["details"][0]["issue"] == "unsupported_extension"


def test_parse_document_large_file(monkeypatch):
    app = create_app()
    with TestClient(app) as client:
        app.state.runtime.settings.api.max_upload_size_mb = 0
        response = client.post(
            "/parse-document",
            files={"file": ("invoice.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 413


def test_parse_document_error_handling(monkeypatch):
    app = create_app()

    with TestClient(app) as client:
        app.state.runtime.model_service = DummyModelService()
        monkeypatch.setattr(
            app.state.runtime.parser_service,
            "parse_file",
            lambda file_path, debug=False: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        response = client.post(
            "/parse-document",
            files={"file": ("invoice.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 500
        assert response.json()["error"] == "Internal processing error."
