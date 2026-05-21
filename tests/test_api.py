from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import MAX_UPLOAD_BYTES
from app.main import app


client = TestClient(app)


def test_health_reports_model_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "ok"
    assert payload["model"]["model_id"] == "mlx-community/gemma-4-e4b-it-8bit"


def test_recognize_rejects_non_image_upload() -> None:
    response = client.post(
        "/api/recognize",
        files={"image": ("frame.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Upload must be an image."


def test_recognize_rejects_empty_upload() -> None:
    response = client.post(
        "/api/recognize",
        files={"image": ("frame.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload is empty."


def test_recognize_rejects_oversized_upload() -> None:
    response = client.post(
        "/api/recognize",
        files={"image": ("frame.jpg", b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image is too large."


def test_recognize_wraps_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_recognize(_path):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.main.model_service.recognize", fail_recognize)

    response = client.post(
        "/api/recognize",
        files={"image": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Recognition failed: model unavailable"


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "blue").save(buffer, format="JPEG")
    return buffer.getvalue()
