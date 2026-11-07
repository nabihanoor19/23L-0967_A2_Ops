import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # must be set before TensorFlow is imported

import json
from pathlib import Path

import tensorflow as tf

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.keras")
META_PATH = os.environ.get("MODEL_META_PATH", "models/champion_meta.json")


class InvalidImageError(ValueError):
    """The uploaded bytes could not be decoded as an image."""


class Predictor:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = int(self.model.input_shape[1])
        meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        self.threshold = float(os.environ.get("THRESHOLD") or meta.get("val_threshold", 0.5))
        self.model_version = str(meta.get("version", "unknown"))
        # compiled forward pass, plus one warm-up call so the first request is not slower
        self._forward = tf.function(lambda x: self.model(x, training=False),
                                    reduce_retracing=True)
        self._forward(tf.zeros((1, self.img_size, self.img_size, 3), dtype=tf.float32))

    def preprocess(self, image_bytes):
        """Same operations as the training pipeline: decode to 3 channels, resize, float32.
        The scaling for MobileNetV2 is the first layer inside the model."""
        try:
            img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            raise InvalidImageError("The file could not be decoded as an image.") from e
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return tf.cast(img, tf.float32)[None, ...]

    def predict(self, image_bytes):
        x = self.preprocess(image_bytes)
        p_ai = float(self._forward(x).numpy()[0][0])
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p_ai if is_ai else 1.0 - p_ai,
                "p_ai": p_ai,
                "threshold": self.threshold,
                "model_version": self.model_version}


_predictor = None


def load_model_once():
    """Called once at start-up. Later calls return the same loaded model."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor