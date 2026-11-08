import json
import os
from pathlib import Path

import requests

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # must be set before TensorFlow is imported

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.keras")
META_PATH = os.environ.get("MODEL_META_PATH", "models/champion_meta.json")
PREDICTOR_URL = os.environ.get("PREDICTOR_URL", "").strip()
PREDICTOR_API_KEY = os.environ.get("PREDICTOR_API_KEY", "").strip()


class InvalidImageError(ValueError):
    """The uploaded bytes could not be decoded as an image."""


class Predictor:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = int(self.model.input_shape[1])
        meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        self.threshold = float(os.environ.get("THRESHOLD") or meta.get("val_threshold", 0.5))
        self.model_version = str(meta.get("version", "unknown"))
        self._tf = tf
        # compiled forward pass, plus one warm-up call so the first request is not slower
        self._forward = tf.function(lambda x: self.model(x, training=False),
                                    reduce_retracing=True)
        self._forward(tf.zeros((1, self.img_size, self.img_size, 3), dtype=tf.float32))

    def preprocess(self, image_bytes):
        """Same operations as the training pipeline: decode to 3 channels, resize, float32.
        The scaling for MobileNetV2 is the first layer inside the model."""
        tf = self._tf
        try:
            img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            raise InvalidImageError("The file could not be decoded as an image.") from e
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return tf.cast(img, tf.float32)[None, ...]

    def predict(self, image_bytes, filename="upload.jpg"):
        x = self.preprocess(image_bytes)
        p_ai = float(self._forward(x).numpy()[0][0])
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p_ai if is_ai else 1.0 - p_ai,
                "p_ai": p_ai,
                "threshold": self.threshold,
                "model_version": self.model_version}


class CloudPredictor:
    """Predictor that delegates inference to the hosted Cloud/Vercel serverless ONNX API."""

    def __init__(self, endpoint_url=None, api_key=None, timeout=30):
        self.endpoint_url = (endpoint_url or os.environ.get("PREDICTOR_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("PREDICTOR_API_KEY", "")
        self.timeout = timeout
        self.threshold = 0.4296
        self.model_version = "2"
        # Query remote health endpoint to verify connection and read model metadata
        try:
            r = requests.get(f"{self.endpoint_url}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                self.threshold = float(data.get("threshold", self.threshold))
                self.model_version = str(data.get("model_version", self.model_version))
        except Exception:
            pass

    def predict(self, image_bytes, filename="upload.jpg"):
        url = f"{self.endpoint_url}/predict"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        files = {"image": (filename, image_bytes, "image/jpeg")}
        try:
            resp = requests.post(url, headers=headers, files=files, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"Could not reach cloud API at {self.endpoint_url}: {e}") from e

        if resp.status_code == 401:
            raise RuntimeError("Cloud API authentication error: invalid or missing API key.")
        if resp.status_code == 413:
            raise ValueError("The cloud API rejected the upload: image file or pixel size is too large.")
        if resp.status_code == 415:
            raise InvalidImageError("The cloud API rejected the image format.")
        if resp.status_code == 422:
            raise InvalidImageError("The cloud API could not decode the uploaded image.")
        if resp.status_code != 200:
            raise RuntimeError(f"Cloud API returned error {resp.status_code}: {resp.text}")

        data = resp.json()
        return {
            "label": data["label"],
            "confidence": float(data["confidence"]),
            "p_ai": float(data["p_ai"]),
            "threshold": float(data["threshold"]),
            "model_version": str(data["model_version"]),
            "cloud_latency_ms": data.get("latency_ms"),
            "backend": "Cloud API (Vercel Serverless ONNX)",
        }


_predictor = None


def load_model_once():
    """Called once at start-up. Returns CloudPredictor when PREDICTOR_URL is configured,
    or falls back to local TensorFlow Predictor."""
    global _predictor
    if _predictor is None:
        url = os.environ.get("PREDICTOR_URL", "").strip()
        if url and not url.startswith("https://replace-after-deploy"):
            api_key = os.environ.get("PREDICTOR_API_KEY", "")
            _predictor = CloudPredictor(endpoint_url=url, api_key=api_key)
        else:
            _predictor = Predictor()
    return _predictor
