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

    def predict(self, tokens):
        _ = tokens
        return [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.97},
            {"text": "Number", "label": "I-KEY", "confidence": 0.96},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.95},
        ]


class DummyRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.model_service = DummyModelService()
        self.model_loaded = True
        self.ocr_loaded = True
        self.startup_error = None


def test_health_endpoint(monkeypatch):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}


def test_parse_document_valid_request(monkeypatch):
    app = create_app()

    monkeypatch.setattr("api.dependencies.process_document", lambda path: [{"text": "Invoice", "confidence": 0.99}])
    monkeypatch.setattr("api.dependencies._count_pages", lambda path: 1)
    monkeypatch.setattr(
        "api.dependencies.postprocess_predictions",
        lambda predictions: {
            "invoice_number": {"value": "INV-1023", "confidence": 0.95, "valid": True},
            "_errors": [],
            "_constraint_flags": [],
        },
    )

    with TestClient(app) as client:
        app.state.runtime.model_service = DummyModelService()
        response = client.post(
            "/parse-document",
            files={"file": ("invoice.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["document"]["invoice_number"]["value"] == "INV-1023"
        assert payload["metadata"]["page_count"] == 1


def test_parse_document_invalid_file():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/parse-document",
            files={"file": ("bad.txt", b"invalid", "text/plain")},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid file format"


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
    monkeypatch.setattr("api.dependencies.process_document", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))

    with TestClient(app) as client:
        app.state.runtime.model_service = DummyModelService()
        response = client.post(
            "/parse-document",
            files={"file": ("invoice.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 500
        assert response.json()["error"] == "Internal processing error."
