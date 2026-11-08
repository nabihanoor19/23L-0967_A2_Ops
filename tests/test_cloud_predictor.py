import sys
from unittest.mock import MagicMock, patch

import pytest

# Remove synthetic stub if test_app.py ran previously
if "src.inference" in sys.modules and getattr(sys.modules["src.inference"], "__file__", None) is None:
    del sys.modules["src.inference"]

from src.inference import CloudPredictor, InvalidImageError  # noqa: E402


def test_cloud_predictor_health_query():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "status": "ok",
        "model_version": "2",
        "threshold": 0.4296,
    }
    with patch("requests.get", return_value=fake_resp):
        cp = CloudPredictor(endpoint_url="https://fake-cloud.vercel.app", api_key="secret-key")
        assert cp.model_version == "2"
        assert cp.threshold == 0.4296
        assert cp.api_key == "secret-key"


def test_cloud_predictor_successful_predict():
    health_resp = MagicMock(status_code=200)
    health_resp.json.return_value = {"model_version": "2", "threshold": 0.4296}

    predict_resp = MagicMock(status_code=200)
    predict_resp.json.return_value = {
        "label": "Real",
        "confidence": 0.85,
        "p_ai": 0.15,
        "threshold": 0.4296,
        "model_version": "2",
        "latency_ms": 28.5,
    }

    with patch("requests.get", return_value=health_resp), \
         patch("requests.post", return_value=predict_resp) as mock_post:
        cp = CloudPredictor(endpoint_url="https://fake-cloud.vercel.app", api_key="test-key")
        result = cp.predict(b"fake-bytes", filename="photo.jpg")

        assert result["label"] == "Real"
        assert result["confidence"] == 0.85
        assert result["p_ai"] == 0.15
        assert result["backend"] == "Cloud API (Vercel Serverless ONNX)"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://fake-cloud.vercel.app/predict"
        assert kwargs["headers"]["X-API-Key"] == "test-key"


def test_cloud_predictor_auth_failure():
    health_resp = MagicMock(status_code=200)
    health_resp.json.return_value = {"model_version": "2", "threshold": 0.4296}

    predict_resp = MagicMock(status_code=401)
    predict_resp.text = "missing or invalid API key"

    with patch("requests.get", return_value=health_resp), \
         patch("requests.post", return_value=predict_resp):
        cp = CloudPredictor(endpoint_url="https://fake-cloud.vercel.app", api_key="bad-key")
        with pytest.raises(RuntimeError, match="authentication error"):
            cp.predict(b"bytes")


def test_cloud_predictor_invalid_image():
    health_resp = MagicMock(status_code=200)
    health_resp.json.return_value = {"model_version": "2", "threshold": 0.4296}

    predict_resp = MagicMock(status_code=422)
    predict_resp.text = "file cannot be decoded as an image"

    with patch("requests.get", return_value=health_resp), \
         patch("requests.post", return_value=predict_resp):
        cp = CloudPredictor(endpoint_url="https://fake-cloud.vercel.app", api_key="key")
        with pytest.raises(InvalidImageError):
            cp.predict(b"corrupt")
