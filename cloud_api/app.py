import hmac
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from PIL import Image

HERE = Path(__file__).parent
META = json.loads((HERE / "model_meta.json").read_text())
THRESHOLD = float(os.environ.get("THRESHOLD") or META["val_threshold"])
MODEL_VERSION = str(META.get("version", "unknown"))
API_KEY = os.environ.get("API_KEY", "")
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "4.4")) * 1024 * 1024)   # Vercel allows 4.5 MB
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "16000000"))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

# loaded once when the function starts, not for every request
session = ort.InferenceSession(str(HERE / "visionguard.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name


def error(message, status):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def too_large(e):
    return error("file too large", 413)


@app.errorhandler(404)
def not_found(e):
    return error("not found", 404)


@app.errorhandler(405)
def not_allowed(e):
    return error("method not allowed", 405)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION,
                    "threshold": round(THRESHOLD, 4)})


@app.post("/predict")
def predict():
    if not API_KEY:
        return error("server is not configured: API_KEY is missing", 503)
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent.encode(), API_KEY.encode()):
        return error("missing or invalid API key", 401)

    upload = request.files.get("image")
    if upload is None:
        return error("send the image as the multipart form field 'image'", 400)
    data = upload.read()
    if not data:
        return error("empty file", 400)

    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                return error("unsupported image format (JPEG, PNG, WEBP, BMP only)", 415)
            if im.width * im.height > MAX_PIXELS:
                return error("image has too many pixels", 413)
            im.load()                         # decodes every pixel: fails for damaged files
            rgb = im.convert("RGB")
    except Exception:
        return error("file cannot be decoded as an image", 422)

    started = time.perf_counter()
    batch = np.asarray(rgb, dtype="float32")[None, ...]          # resizing is inside the model
    p_ai = float(session.run(None, {INPUT_NAME: batch})[0].ravel()[0])
    latency_ms = (time.perf_counter() - started) * 1000
    is_ai = p_ai >= THRESHOLD
    return jsonify({
        "label": "AI-generated" if is_ai else "Real",
        "confidence": round(p_ai if is_ai else 1.0 - p_ai, 4),
        "p_ai": round(p_ai, 4),
        "threshold": round(THRESHOLD, 4),
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency_ms, 1),
    })


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5050")), debug=False)