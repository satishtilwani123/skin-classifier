"""
tests/test_api.py

Full API test suite for the Skin Disease Classifier.
Covers: /predict happy path, validation errors, wrong file types,
        corrupted data, preprocessor logic, explainer logic, and
        the model singleton. Designed to run in CI/CD without a
        GPU — the model is mocked so TensorFlow is never loaded.

Run locally:
    pytest tests/ -v

Run in CI (see .github/workflows/main.yml):
    pytest tests/ -v --tb=short
"""

import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes(width: int = 100, height: int = 100, color=(120, 80, 60)) -> bytes:
    """Create a minimal in-memory JPEG image and return its raw bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal in-memory PNG image."""
    img = Image.new("RGB", (width, height), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_grayscale_bytes() -> bytes:
    """Create a grayscale (L-mode) PNG — tests the 2-D→3-D channel fix."""
    img = Image.new("L", (50, 50), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# App fixture — model is mocked so TF is never imported
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Build the FastAPI app with the ML model mocked.

    The mock model returns a fixed prediction: [0.8, 0.2]
    (80 % benign, 20 % malignant). This lets every test run
    without TensorFlow or a real .h5 file.
    """
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
# 1. Happy path — real image uploads
# ---------------------------------------------------------------------------

class TestPredictHappyPath:

    def test_jpeg_image_returns_200(self, client):
        """Standard JPEG upload should return HTTP 200."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_png_image_returns_200(self, client):
        """PNG images should also be accepted."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", make_png_bytes(), "image/png")},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        """Response JSON must contain label, class_index, confidence, advice."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        body = response.json()
        assert "label"       in body, "Missing 'label' field"
        assert "class_index" in body, "Missing 'class_index' field"
        assert "confidence"  in body, "Missing 'confidence' field"
        assert "advice"      in body, "Missing 'advice' field"

    def test_confidence_is_between_0_and_1(self, client):
        """Confidence value must be a probability in [0, 1]."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"

    def test_class_index_is_valid(self, client):
        """class_index must be 0 (benign) or 1 (malignant)."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert response.json()["class_index"] in (0, 1)

    def test_label_is_string(self, client):
        """label must be a non-empty string."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        label = response.json()["label"]
        assert isinstance(label, str) and len(label) > 0

    def test_advice_is_string(self, client):
        """advice must be a non-empty string."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        advice = response.json()["advice"]
        assert isinstance(advice, str) and len(advice) > 0

    def test_isic_image_returns_200(self, client):
        """Upload the real ISIC dermoscopy image used during development."""
        import os
        image_path = os.path.join(os.path.dirname(__file__), "..", "ISIC_0029306.jpg")
        if not os.path.exists(image_path):
            pytest.skip("ISIC_0029306.jpg not found — skipping real-image test")
        with open(image_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("ISIC_0029306.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 200

    def test_isic_image_response_structure(self, client):
        """Real ISIC image response must have all required fields."""
        import os
        image_path = os.path.join(os.path.dirname(__file__), "..", "ISIC_0029306.jpg")
        if not os.path.exists(image_path):
            pytest.skip("ISIC_0029306.jpg not found")
        with open(image_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("ISIC_0029306.jpg", f, "image/jpeg")},
            )
        body = response.json()
        for field in ("label", "class_index", "confidence", "advice"):
            assert field in body


# ---------------------------------------------------------------------------
# 2. Validation errors (expect 400)
# ---------------------------------------------------------------------------

class TestPredictValidationErrors:

    def test_no_file_returns_422(self, client):
        """Calling /predict with no file should return HTTP 422 (unprocessable)."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_non_image_content_type_returns_400(self, client):
        """Uploading a plain text file should be rejected with 400."""
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    def test_pdf_file_returns_400(self, client):
        """PDFs are not images — should return 400."""
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_empty_file_returns_400(self, client):
        """An empty file body should be rejected."""
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_corrupted_image_returns_400(self, client):
        """Random bytes with image/* content type should return 400."""
        response = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"\x00\x01\x02\x03NOTANIMAGE", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_json_body_returns_422(self, client):
        """Sending JSON instead of a file upload should return 422."""
        response = client.post(
            "/predict",
            json={"image": "base64data"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Edge-case images
# ---------------------------------------------------------------------------

class TestEdgeCaseImages:

    def test_very_small_image_accepted(self, client):
        """A 1×1 pixel image should still be resized and processed."""
        tiny = make_jpeg_bytes(width=1, height=1)
        response = client.post(
            "/predict",
            files={"file": ("tiny.jpg", tiny, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_large_image_accepted(self, client):
        """A large 1024×1024 image should be resized down and accepted."""
        large = make_jpeg_bytes(width=1024, height=1024)
        response = client.post(
            "/predict",
            files={"file": ("large.jpg", large, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_non_square_image_accepted(self, client):
        """Portrait aspect ratio images should be accepted and resized."""
        portrait = make_jpeg_bytes(width=300, height=600)
        response = client.post(
            "/predict",
            files={"file": ("portrait.jpg", portrait, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_grayscale_image_accepted(self, client):
        """Grayscale (single-channel) images should be converted to RGB."""
        gray = make_grayscale_bytes()
        response = client.post(
            "/predict",
            files={"file": ("gray.png", gray, "image/png")},
        )
        # preprocessor converts L-mode to RGB, so this should work
        assert response.status_code == 200

    def test_all_black_image_accepted(self, client):
        """A completely black image is valid input."""
        black = make_jpeg_bytes(color=(0, 0, 0))
        response = client.post(
            "/predict",
            files={"file": ("black.jpg", black, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_all_white_image_accepted(self, client):
        """A completely white image is valid input."""
        white = make_jpeg_bytes(color=(255, 255, 255))
        response = client.post(
            "/predict",
            files={"file": ("white.jpg", white, "image/jpeg")},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 4. Preprocessor unit tests (no HTTP, pure logic)
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def setup_method(self):
        from app_backend.config import Settings
        self.settings = Settings()

    def test_output_is_4d(self):
        """preprocess_bytes must return a 4-D array (batch, H, W, C)."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        assert batch.ndim == 4, f"Expected 4D, got {batch.ndim}D shape {batch.shape}"

    def test_output_shape_matches_input_size(self):
        """H and W must equal settings.input_size."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        _, h, w, _ = batch.shape
        assert h == self.settings.input_size, f"Height {h} != input_size {self.settings.input_size}"
        assert w == self.settings.input_size, f"Width  {w} != input_size {self.settings.input_size}"

    def test_output_has_3_channels(self):
        """Output must always have exactly 3 channels (RGB)."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        assert batch.shape[-1] == 3, f"Expected 3 channels, got {batch.shape[-1]}"

    def test_pixel_values_normalised_0_to_1(self):
        """All pixel values must be in [0, 1] after /255 normalisation."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        assert batch.min() >= 0.0, f"Pixel min below 0: {batch.min()}"
        assert batch.max() <= 1.0, f"Pixel max above 1: {batch.max()}"

    def test_batch_size_is_1(self):
        """Batch dimension must be exactly 1."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        assert batch.shape[0] == 1

    def test_grayscale_converted_to_3_channels(self):
        """Grayscale PNG must be converted to 3-channel output."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_grayscale_bytes(), self.settings)
        assert batch.shape[-1] == 3

    def test_corrupted_bytes_raises_http_400(self):
        """Corrupted image bytes must raise HTTPException with status 400."""
        from app_backend.services.preprocessor import preprocess_bytes
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            preprocess_bytes(b"\x00\x01GARBAGE", self.settings)
        assert exc_info.value.status_code == 400

    def test_dtype_is_float32(self):
        """Output array dtype must be float32 for TensorFlow compatibility."""
        from app_backend.services.preprocessor import preprocess_bytes
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        assert batch.dtype == np.float32, f"Expected float32, got {batch.dtype}"

    def test_validate_shape_passes_for_correct_batch(self):
        """validate_shape must not raise for a correctly shaped batch."""
        from app_backend.services.preprocessor import preprocess_bytes, validate_shape
        batch = preprocess_bytes(make_jpeg_bytes(), self.settings)
        shape = validate_shape(batch, self.settings)
        assert len(shape) == 4

    def test_validate_shape_raises_for_wrong_size(self):
        """validate_shape must raise ValueError if H/W don't match settings."""
        from app_backend.services.preprocessor import validate_shape
        wrong = np.zeros((1, 64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            validate_shape(wrong, self.settings)

    def test_validate_shape_raises_for_3d_input(self):
        """validate_shape must raise ValueError for a 3-D (non-batched) array."""
        from app_backend.services.preprocessor import validate_shape
        bad = np.zeros((28, 28, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            validate_shape(bad, self.settings)


# ---------------------------------------------------------------------------
# 5. Explainer unit tests
# ---------------------------------------------------------------------------

class TestExplainer:

    def setup_method(self):
        from app_backend.config import Settings
        self.settings = Settings()

    def test_benign_label_returned_for_index_0(self):
        from app_backend.services.explainer import explain
        result = explain(0, 0.9, self.settings)
        assert result["label"] == "benign"

    def test_malignant_label_returned_for_index_1(self):
        from app_backend.services.explainer import explain
        result = explain(1, 0.7, self.settings)
        assert result["label"] == "malignant"

    def test_benign_advice_mentions_low_risk(self):
        from app_backend.services.explainer import explain
        result = explain(0, 0.9, self.settings)
        assert "low risk" in result["advice"].lower() or "monitor" in result["advice"].lower()

    def test_malignant_advice_mentions_dermatologist(self):
        from app_backend.services.explainer import explain
        result = explain(1, 0.95, self.settings)
        assert "dermatologist" in result["advice"].lower() or "risk" in result["advice"].lower()

    def test_result_always_has_label_and_advice_keys(self):
        from app_backend.services.explainer import explain
        for idx in (0, 1):
            result = explain(idx, 0.5, self.settings)
            assert "label" in result
            assert "advice" in result

    def test_out_of_range_index_returns_unknown(self):
        from app_backend.services.explainer import explain
        result = explain(99, 0.5, self.settings)
        assert result["label"] == "unknown"


# ---------------------------------------------------------------------------
# 6. Settings / config tests
# ---------------------------------------------------------------------------

class TestSettings:

    def test_default_input_size_is_28(self):
        from app_backend.config import Settings
        s = Settings()
        assert s.input_size == 28

    def test_default_class_names(self):
        from app_backend.config import Settings
        s = Settings()
        assert "benign" in s.class_names
        assert "malignant" in s.class_names

    def test_model_path_is_path_object(self):
        from app_backend.config import Settings
        from pathlib import Path
        s = Settings()
        assert isinstance(s.model_path, Path)


# ---------------------------------------------------------------------------
# 7. Health / routing smoke tests
# ---------------------------------------------------------------------------

class TestRouting:

    def test_root_returns_404(self, client):
        """There is no root route — should return 404."""
        response = client.get("/")
        assert response.status_code == 404

    def test_docs_are_accessible(self, client):
        """FastAPI's /docs page must be reachable."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_is_accessible(self, client):
        """/openapi.json must return a valid spec."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "openapi" in spec
        assert "paths" in spec

    def test_predict_endpoint_exists_in_openapi(self, client):
        """/predict must be listed in the OpenAPI spec."""
        spec = client.get("/openapi.json").json()
        assert "/predict" in spec["paths"]

    def test_predict_only_accepts_post(self, client):
        """GET to /predict should return 405 Method Not Allowed."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_predict_put_not_allowed(self, client):
        """PUT to /predict should return 405."""
        response = client.put("/predict")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# 8. Model singleton tests
# ---------------------------------------------------------------------------

class TestModelSingleton:

    def test_get_model_raises_if_not_initialized(self):
        """get_model must raise RuntimeError when _model is None and no settings given."""
        import app_backend.models.ml_models as mm
        original = mm._model
        try:
            mm._model = None
            with pytest.raises((RuntimeError, Exception)):
                mm.get_model(settings=None)
        finally:
            mm._model = original

    def test_get_model_returns_same_instance(self):
        """get_model must return the same singleton object on repeated calls."""
        import app_backend.models.ml_models as mm
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.6, 0.4]])
        original = mm._model
        try:
            mm._model = mock
            assert mm.get_model() is mock
            assert mm.get_model() is mock
        finally:
            mm._model = original
