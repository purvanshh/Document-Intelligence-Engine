from __future__ import annotations

import pytest

from api.schemas import ErrorResponse, ParseDocumentResponse


@pytest.mark.asyncio
async def test_health_endpoint(api_client):
    client, _ = api_client
    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_loaded" in payload


@pytest.mark.asyncio
async def test_parse_document_valid_upload(api_client, sample_image_bytes, mock_structured_output, monkeypatch):
    client, app = api_client

    monkeypatch.setattr(
        app.state.runtime.parser_service,
        "parse_file",
        lambda file_path, debug=False: {
            "document": mock_structured_output,
            "metadata": {
                "filename": "sample.png",
                "page_count": 1,
                "ocr_token_count": 8,
                "confidence_summary": {
                    "average": 0.951667,
                    "minimum": 0.93,
                    "maximum": 0.97,
                    "kept_fields": 3,
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
                "model": {
                    "name": "microsoft/layoutlmv3-base",
                    "version": "0.1.0",
                    "device": "cpu",
                },
            },
        },
    )

    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    ParseDocumentResponse.model_validate(response.json())


@pytest.mark.asyncio
async def test_parse_document_invalid_file_type(api_client):
    client, _ = api_client
    response = await client.post(
        "/parse-document",
        files={"file": ("sample.txt", b"bad", "text/plain")},
    )
    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Invalid file format"


@pytest.mark.asyncio
async def test_parse_document_large_file(api_client, sample_image_bytes):
    client, app = api_client
    app.state.runtime.settings.api.max_upload_size_mb = 0
    response = await client.post(
        "/parse-document",
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 413
    payload = ErrorResponse.model_validate(response.json())
    assert payload.code == 413


@pytest.mark.asyncio
async def test_parse_document_empty_file(api_client):
    client, _ = api_client
    response = await client.post(
        "/parse-document",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    payload = ErrorResponse.model_validate(response.json())
    assert payload.error == "Uploaded file is empty."
