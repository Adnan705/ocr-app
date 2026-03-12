"""
app.py
======
Flask web server for the OCR pipeline.
Exposes:
  GET  /            – Serve the web UI
  POST /ocr         – Accept an uploaded image, return extracted text (JSON)
  GET  /health      – Health-check endpoint for Railway
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image as PILImage
from werkzeug.utils import secure_filename

# ── OCR pipeline modules ──────────────────────────────────────────────────────
from preprocessing import preprocess
from detection     import detect
from recognition   import recognize
from utils         import clean_text, merge_texts, has_table

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16 MB upload cap
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Railway health-check."""
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    """Serve the main UI page."""
    return send_from_directory("static", "index.html")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    POST /ocr
    ---------
    Body     : multipart/form-data with field `image`
    Optional : form fields
                  detect_method  – auto | east | contour  (default: contour)
                  ocr_backend    – auto | easyocr | tesseract  (default: auto)
                  languages      – comma-separated EasyOCR codes  (default: en)
    Returns  : JSON
                  { text, regions, has_table, elapsed_ms, error? }
    """
    import time

    # ── Validate upload ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXTENSIONS}"}), 400

    # ── Read options ──────────────────────────────────────────────────────────
    detect_method = request.form.get("detect_method", "contour")
    ocr_backend   = request.form.get("ocr_backend",   "auto")
    languages     = [l.strip() for l in
                     request.form.get("languages", "en").split(",")]

    # ── Save upload to a temp file ────────────────────────────────────────────
    suffix = Path(secure_filename(file.filename)).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        t0 = time.time()

        # 1. Preprocessing
        original_bgr, _ = preprocess(
            tmp_path,
            denoise_method  = "gaussian",   # faster for web use
            binarize_method = "otsu",
            do_deskew       = True,
        )

        # 2. Detection
        boxes, crops, _ = detect(original_bgr, method=detect_method)

        # 3. Recognition
        raw_texts = recognize(crops, backend=ocr_backend, languages=languages)

        # 4. Postprocessing
        cleaned = [clean_text(t) for t in raw_texts]
        full_text = merge_texts(cleaned)

        # 5. Table detection
        table_found = has_table(original_bgr)

        elapsed_ms = int((time.time() - t0) * 1000)

        return jsonify({
            "text":       full_text,
            "regions":    len(boxes),
            "has_table":  table_found,
            "elapsed_ms": elapsed_ms,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        os.unlink(tmp_path)   # always clean up the temp file


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting OCR server on port {port} …")
    app.run(host="0.0.0.0", port=port, debug=debug)
