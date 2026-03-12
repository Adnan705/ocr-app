# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — OCR Web App
# Base: Python 3.11 slim + system Tesseract + OpenCV deps
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0  → OpenCV runtime
# tesseract-ocr          → Tesseract OCR backend
# libgomp1               → OpenMP (used by some torch ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      tesseract-ocr \
      tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches pip install
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller image than CUDA variant)
RUN pip install --no-cache-dir \
      torch==2.2.0+cpu \
      torchvision==0.17.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py           .
COPY preprocessing.py .
COPY detection.py     .
COPY recognition.py   .
COPY utils.py         .
COPY static/          ./static/

# ── Create writable dirs ──────────────────────────────────────────────────────
RUN mkdir -p /app/output /app/models /tmp/ocr

# ── Pre-download EasyOCR English model at build time ─────────────────────────
# This avoids a 100 MB download on the first user request.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" || true

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Start server ──────────────────────────────────────────────────────────────
# gunicorn: production WSGI server (more robust than Flask dev server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "app:app"]
