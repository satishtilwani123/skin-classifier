"""
tests/test_api.py
 
Skin Disease Classifier — API test suite.
Runs in CI/CD without a GPU; the ML model is mocked.
 
Run locally:
    pytest tests/ -v --tb=short
"""
 
import io
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
 
 
# ---------------------------------------------------------------------------
# App fixture — ML model mocked, returns benign (0.8) / malignant (0.2)
# ---------------------------------------------------------------------------
 
@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2]])
 
    with patch("app_backend.models.ml_models._model", mock_model), \
         patch("app_backend.models.ml_models.initialize", return_value=mock_model), \
         patch("app_backend.models.ml_models.get_model", return_value=mock_model):
 
        from app_backend.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c
 
 
# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------
 
def test_predict_jpeg_returns_200(client):
    """JPEG upload must return HTTP 200."""
    response = client.post(
        "/predict",
        files={"file": ("skin.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
 
 
def test_predict_response_has_all_fields(client):
    """Response must contain label, class_index, confidence, and advice."""
    response = client.post(
        "/predict",
        files={"file": ("skin.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    body = response.json()
    for field in ("label", "class_index", "confidence", "advice"):
        assert field in body, f"Missing field: '{field}'"
 
 
def test_predict_confidence_range(client):
    """Confidence must be a valid probability between 0 and 1."""
    response = client.post(
        "/predict",
        files={"file": ("skin.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
 
 
def test_predict_class_index_valid(client):
    """class_index must be 0 (benign) or 1 (malignant)."""
    response = client.post(
        "/predict",
        files={"file": ("skin.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.json()["class_index"] in (0, 1)
 
 
def test_predict_no_file_returns_422(client):
    """Calling /predict with no file must return HTTP 422."""
    response = client.post("/predict")
    assert response.status_code == 422
 
 
def test_predict_non_image_returns_400(client):
    """Uploading a text file must be rejected with HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
 
 
def test_predict_corrupted_image_returns_400(client):
    """Random bytes with image/* content type must return HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\x00\x01\x02NOTANIMAGE", "image/jpeg")},
    )
    assert response.status_code == 400
 
 
def test_predict_get_method_not_allowed(client):
    """GET on /predict must return HTTP 405."""
    response = client.get("/predict")
    assert response.status_code == 405
 
 
def test_predict_isic_real_image(client):
    """Real ISIC dermoscopy image must return 200 with all response fields."""
    image_path = os.path.join(os.path.dirname(__file__), "..", "ISIC_0029306.jpg")
    if not os.path.exists(image_path):
        pytest.skip("ISIC_0029306.jpg not present — skipping")
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("ISIC_0029306.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    for field in ("label", "class_index", "confidence", "advice"):
        assert field in body
 
 
def test_docs_endpoint_accessible(client):
    """FastAPI /docs must be reachable — confirms the app booted correctly."""
    response = client.get("/docs")
    assert response.status_code == 200