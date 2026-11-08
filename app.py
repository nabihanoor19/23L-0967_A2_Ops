import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # must be set before TensorFlow is imported

from dotenv import load_dotenv

load_dotenv()   # read .env first: src.db and src.inference read their settings when imported

import base64
import io
import time

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src import db
from src.inference import InvalidImageError, load_model_once

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "30000000"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)

db.init_db()
predictor = load_model_once()            # loaded once, at start-up


class UserError(Exception):
    def __init__(self, title, message, status=400):
        self.title, self.message, self.status = title, message, status


def inspect_image(data):
    """Confirm the bytes are a real, fully readable image. Returns width, height and a
    small preview (base64 JPEG built in memory)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise UserError("Image format not supported",
                                "VisionGuard accepts JPEG, PNG, WEBP and BMP images only.", 415)
            width, height = im.size
            if width * height > MAX_PIXELS:
                raise UserError("Image has too many pixels",
                                f"Images above {MAX_PIXELS / 1e6:.0f} megapixels are not accepted.", 413)
            im.load()                    # decodes every pixel: fails for truncated or corrupt files
            preview = im.convert("RGB")
            preview.thumbnail((480, 480))
            buf = io.BytesIO()
            preview.save(buf, format="JPEG", quality=80)     # buf is memory, not a file
    except UserError:
        raise
    except Exception as e:               # any decoder failure
        raise UserError("This file cannot be read as an image",
                        "It looks like an image, but it is damaged or incomplete. "
                        "Please try a different file.", 422) from e
    return width, height, base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise UserError("No file chosen", "Please choose an image before pressing the button.", 400)

    extension = os.path.splitext(upload.filename)[1].lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise UserError("File type not allowed",
                        "VisionGuard accepts JPG, PNG, WEBP and BMP images only.", 415)
    safe_name = (secure_filename(upload.filename) or f"upload.{extension}")[:100]

    data = upload.read()                 # kept in memory only, never written by this code
    if not data:
        raise UserError("Empty file", "The file you sent has no content.", 400)

    width, height, preview = inspect_image(data)

    started = time.perf_counter()
    try:
        result = predictor.predict(data)
    except InvalidImageError as e:
        raise UserError("This file cannot be read as an image",
                        "The model could not decode this file.", 422) from e
    except ValueError as e:
        raise UserError("Image rejected", str(e), 413) from e
    except RuntimeError as e:
        raise UserError("Prediction failed", str(e), 502) from e
    latency_ms = (time.perf_counter() - started) * 1000

    db.save_prediction(filename=safe_name, file_size_kb=len(data) / 1024,
                       label=result["label"], confidence=result["confidence"],
                       p_ai=result["p_ai"], threshold=result["threshold"],
                       model_version=result["model_version"], latency_ms=latency_ms)
    return render_template("result.html", result=result, filename=safe_name, width=width,
                           height=height, size_kb=len(data) / 1024, preview=preview,
                           latency_ms=latency_ms,
                           borderline=abs(result["p_ai"] - result["threshold"]) < 0.10)


@app.get("/history")
def history():
    return render_template("history.html", rows=db.recent_predictions(100))


@app.get("/health")
def health():
    database_ok = db.db_ok()
    backend_type = "cloud_api" if predictor.__class__.__name__ == "CloudPredictor" else "local_model"
    body = {"status": "ok" if database_ok else "degraded",
            "model_loaded": True,
            "model_version": predictor.model_version,
            "threshold": round(predictor.threshold, 4),
            "backend": backend_type,
            "database": "ok" if database_ok else "unreachable",
            "debug_mode": app.debug,
            "max_upload_mb": MAX_UPLOAD_MB}
    return jsonify(body), (200 if database_ok else 503)


def _error_page(title, message, status):
    return render_template("error.html", title=title, message=message, status=status), status


@app.errorhandler(UserError)
def handle_user_error(e):
    return _error_page(e.title, e.message, e.status)


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return _error_page("File too large",
                       f"The upload is bigger than the limit of {MAX_UPLOAD_MB:g} MB. "
                       "Please choose a smaller file.", 413)


@app.errorhandler(404)
def handle_not_found(e):
    return _error_page("Page not found", "This page does not exist.", 404)


@app.errorhandler(405)
def handle_not_allowed(e):
    return _error_page("Method not allowed", "This page cannot be opened this way.", 405)


@app.errorhandler(500)
def handle_server_error(e):
    return _error_page("Something went wrong",
                       "An unexpected error happened on our side. Please try again.", 500)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")), debug=False)
