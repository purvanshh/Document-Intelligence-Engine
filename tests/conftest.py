from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from PIL import Image

from api.main import create_app
from document_intelligence_engine.core.config import get_settings


class FakeOCREngine:
    def __init__(self, pages: list[list[dict[str, object]]]) -> None:
        self._pages = pages
        self._cursor = 0

    def extract_tokens(self, image: Image.Image) -> list[dict[str, object]]:
        _ = image
        if self._cursor >= len(self._pages):
            return []
        result = self._pages[self._cursor]
        self._cursor += 1
        return result

    def extract_batch_tokens(self, images: list[Image.Image]) -> list[list[dict[str, object]]]:
        return [self.extract_tokens(image) for image in images]


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def sample_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_image_bytes: bytes) -> Path:
    path = tmp_path / "sample.png"
    path.write_bytes(sample_image_bytes)
    return path


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), color="white").save(buffer, format="PDF")
    return buffer.getvalue()


@pytest.fixture
def sample_pdf_path(tmp_path: Path, sample_pdf_bytes: bytes) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(sample_pdf_bytes)
    return path


@pytest.fixture
def corrupted_file_path(tmp_path: Path) -> Path:
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not-a-valid-image")
    return path


@pytest.fixture
def mock_ocr_output() -> list[dict[str, object]]:
    return [
        {"text": "Invoice", "bbox": [10, 10, 80, 30], "confidence": 0.99},
        {"text": "Number", "bbox": [90, 10, 170, 30], "confidence": 0.98},
        {"text": "INV-1023", "bbox": [180, 10, 280, 30], "confidence": 0.97},
        {"text": "Date", "bbox": [10, 50, 60, 70], "confidence": 0.96},
        {"text": "2025-01-12", "bbox": [70, 50, 170, 70], "confidence": 0.95},
        {"text": "Total", "bbox": [10, 90, 70, 110], "confidence": 0.94},
        {"text": "Amount", "bbox": [80, 90, 160, 110], "confidence": 0.93},
        {"text": "1200.50", "bbox": [170, 90, 260, 110], "confidence": 0.92},
    ]


@pytest.fixture
def mock_model_output() -> list[dict[str, object]]:
    return [
        {"text": "Invoice", "label": "B-KEY", "confidence": 0.99},
        {"text": "Number", "label": "I-KEY", "confidence": 0.98},
        {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.97},
        {"text": "Date", "label": "B-KEY", "confidence": 0.96},
        {"text": "2025-01-12", "label": "B-VALUE", "confidence": 0.95},
        {"text": "Total", "label": "B-KEY", "confidence": 0.94},
        {"text": "Amount", "label": "I-KEY", "confidence": 0.93},
        {"text": "1200.50", "label": "B-VALUE", "confidence": 0.92},
    ]


@pytest.fixture
def mock_structured_output() -> dict[str, object]:
    return {
        "invoice_number": {"value": "INV-1023", "confidence": 0.97, "valid": True},
        "date": {"value": "2025-01-12", "confidence": 0.955, "valid": True},
        "total_amount": {"value": 1200.5, "confidence": 0.93, "valid": True},
        "_errors": [],
        "_constraint_flags": [],
    }


@pytest_asyncio.fixture
async def api_client():
    app = create_app()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app
