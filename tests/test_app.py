import io
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
from PIL import Image

os.environ["DATABASE_URL"] = "sqlite:///" + (Path(tempfile.mkdtemp()) / "ci_test.db").as_posix()


class InvalidImageError(ValueError):
    pass


class FakePredictor:
    model_version = "test"
    threshold = 0.43

    def predict(self, image_bytes):
        return {"label": "Real", "confidence": 0.8, "p_ai": 0.2,
                "threshold": 0.43, "model_version": "test"}


# The real model is not needed (and TensorFlow is not installed) in CI: use a stub.
_stub = types.ModuleType("src.inference")
_stub.InvalidImageError = InvalidImageError
_stub.load_model_once = lambda: FakePredictor()
sys.modules["src.inference"] = _stub

import app as webapp


@pytest.fixture()
def client():
    return webapp.app.test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def post(client, name, data):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["debug_mode"] is False


def test_upload_page(client):
    assert b"VisionGuard" in client.get("/").data


def test_wrong_extension_is_rejected(client):
    assert post(client, "notes.txt", b"hello").status_code == 415


def test_text_pretending_to_be_image_is_rejected(client):
    assert post(client, "fake.jpg", b"not really an image").status_code == 422


def test_valid_image_is_classified_and_logged(client):
    r = post(client, "red square.png", png_bytes())
    assert r.status_code == 200
    assert b"Real" in r.data
    assert b"red_square.png" in client.get("/history").data


def test_file_name_is_sanitised(client):
    post(client, "../../evil name.png", png_bytes())
    assert b"evil_name.png" in client.get("/history").data


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setitem(webapp.app.config, "MAX_CONTENT_LENGTH", 1000)
    assert post(client, "big.png", png_bytes() + b"0" * 5000).status_code == 413