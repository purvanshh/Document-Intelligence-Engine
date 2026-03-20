from __future__ import annotations

from fastapi.testclient import TestClient

from document_intelligence_engine.api.app import create_app


def test_healthcheck() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
