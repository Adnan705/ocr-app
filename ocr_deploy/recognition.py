"""
recognition.py
==============
Text recognition (OCR) for the pipeline.

Three recognition back-ends are provided:

  1. EasyOCR   – Best quality. Uses CRAFT detector + CRNN (CNN+RNN+CTC)
                 recognition network. Supports 80+ languages. No paid API.
                 Models are downloaded automatically on first run (~100 MB).

  2. Tesseract – Classic open-source OCR engine from Google. Requires the
                 `tesseract` binary to be installed on the system.
                 Install: `sudo apt install tesseract-ocr` (Linux)
                          `brew install tesseract`          (macOS)
                          Download installer from UB Mannheim (Windows)

  3. Custom CNN+RNN+CTC model  – Lightweight PyTorch model you can train on
                                  your own dataset.  Defined in this file so
                                  you can swap in custom weights easily.
                                  Falls back gracefully if PyTorch is absent.

The pipeline tries back-ends in order: EasyOCR → Tesseract → Custom model.
You can also force a specific back-end via the `backend` parameter.
"""

from __future__ import annotations

import re
import sys
from typing import List

import cv2
import numpy as np

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EasyOCR back-end
# ─────────────────────────────────────────────────────────────────────────────

# Keep a single reader instance so models are not reloaded on every call.
_easyocr_reader = None


def _get_easyocr_reader(languages: list[str] | None = None):
    """Return a cached EasyOCR reader; create it on first call."""
    global _easyocr_reader
    if _easyocr_reader is None:
        langs = languages or ["en"]
        print(f"[Recognition] Initialising EasyOCR reader (languages={langs}). "
              "Model files are downloaded on first run (~100 MB) …")
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
        print("[Recognition] EasyOCR reader ready.")
    return _easyocr_reader


def recognize_easyocr(crops: list[np.ndarray],
                      languages: list[str] | None = None) -> list[str]:
    """
    Recognise text in a list of image crops using EasyOCR.

    Args:
        crops:     List of BGR image crops (one per text region).
        languages: EasyOCR language codes, e.g. ["en", "ar"].

    Returns:
        List of recognised strings, one per crop.
    """
    reader  = _get_easyocr_reader(languages)
    results = []

    for i, crop in enumerate(crops):
        # EasyOCR expects RGB or BGR numpy arrays
        output = reader.readtext(crop, detail=0, paragraph=True)
        text   = " ".join(output).strip()
        results.append(text)
        print(f"[Recognition] EasyOCR crop {i+1}/{len(crops)}: '{text}'")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tesseract back-end
# ─────────────────────────────────────────────────────────────────────────────

def recognize_tesseract(crops: list[np.ndarray],
                        lang: str = "eng",
                        config: str = "--psm 6") -> list[str]:
    """
    Recognise text using Tesseract OCR.

    Args:
        crops:  List of BGR image crops.
        lang:   Tesseract language code(s), e.g. "eng" or "eng+ara".
        config: Tesseract page-segmentation mode and other flags.
                PSM 6 = "assume a uniform block of text" (good default).

    Returns:
        List of recognised strings.
    """
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is not installed.  Run:  pip install pytesseract\n"
            "Also install the Tesseract binary for your OS."
        )

    results = []
    for i, crop in enumerate(crops):
        # Convert BGR → RGB PIL image
        pil_img = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        text    = pytesseract.image_to_string(pil_img, lang=lang,
                                              config=config).strip()
        results.append(text)
        print(f"[Recognition] Tesseract crop {i+1}/{len(crops)}: '{text}'")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Custom CNN + RNN + CTC model (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

# Character set used by the custom model
CUSTOM_CHARSET = (
    " !\"#$%&'()*+,-./"
    "0123456789"
    ":;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
)
# Index 0 is reserved for the CTC blank token
BLANK_IDX = 0
NUM_CLASSES = len(CUSTOM_CHARSET) + 1  # +1 for blank


def _char_to_idx(c: str) -> int:
    try:
        return CUSTOM_CHARSET.index(c) + 1  # +1 because 0 = blank
    except ValueError:
        return BLANK_IDX


def _idx_to_char(i: int) -> str:
    if i == BLANK_IDX or i > len(CUSTOM_CHARSET):
        return ""
    return CUSTOM_CHARSET[i - 1]


def _ctc_decode(log_probs: "torch.Tensor") -> str:
    """
    Greedy CTC decode: argmax at each time step → collapse repeats → remove blanks.

    Args:
        log_probs: Tensor of shape (T, num_classes) – log-softmax output.

    Returns:
        Decoded string.
    """
    indices = log_probs.argmax(dim=-1).tolist()

    # Collapse consecutive identical indices, then remove blanks
    chars = []
    prev  = None
    for idx in indices:
        if idx != prev:
            chars.append(_idx_to_char(idx))
        prev = idx

    return "".join(chars)


class _CRNN(nn.Module):
    """
    A small CNN + Bi-LSTM + FC OCR recognition network.

    Architecture summary
    --------------------
    Input  : (B, 1, H, W) grayscale image, H=32
    CNN    : 4 × Conv-BN-ReLU-MaxPool blocks → (B, 512, 1, W')
    Reshape: (W', B, 512)  –  treat width steps as time steps
    BiLSTM : 2 layers  →  (W', B, 512)
    FC     : (W', B, num_classes)
    Output : log-softmax probabilities for CTC loss / decoding
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── CNN feature extractor ────────────────────────────────────────────
        def _conv_block(in_ch, out_ch, pool=(2, 2)):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(
            _conv_block(1,   64,  (2, 2)),   # H/2,  W/2
            _conv_block(64,  128, (2, 2)),   # H/4,  W/4
            _conv_block(128, 256, (2, 1)),   # H/8,  W/4
            _conv_block(256, 512, (2, 1)),   # H/16, W/4  (≈ 2 rows → 1 after pool)
        )

        # ── Sequence model ───────────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=False,
            bidirectional=True,
        )

        # ── Classifier ───────────────────────────────────────────────────────
        self.fc = nn.Linear(512, num_classes)  # 256 * 2 = 512 (bidirectional)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, 1, 32, W) input tensor.

        Returns:
            log-softmax tensor of shape (T, B, num_classes) for CTC.
        """
        # CNN: (B, 1, 32, W) → (B, 512, 2, W')
        features = self.cnn(x)

        # Collapse height dimension: (B, 512, H', W') → (B, 512*H', W')
        b, c, h, w = features.size()
        features = features.view(b, c * h, w)          # (B, C', W')
        features = features.permute(2, 0, 1)           # (W', B, C')

        # RNN
        rnn_out, _ = self.rnn(features)                # (T, B, 512)

        # FC + log-softmax
        logits = self.fc(rnn_out)                      # (T, B, num_classes)
        return nn.functional.log_softmax(logits, dim=2)


def _load_custom_model(weights_path: str | None = None) -> "_CRNN | None":
    """
    Load the CRNN model.  If `weights_path` is provided, load saved weights.
    Otherwise return the model with random (untrained) weights – useful as a
    placeholder or starting point for fine-tuning.
    """
    if not _TORCH_AVAILABLE:
        return None

    model = _CRNN(num_classes=NUM_CLASSES)
    model.eval()

    if weights_path:
        try:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"[Recognition] Custom model weights loaded from: {weights_path}")
        except Exception as exc:
            print(f"[Recognition] Could not load weights ({exc}). "
                  "Using un-trained model.")
    else:
        print("[Recognition] No weights path given – custom model is untrained. "
              "Results will be meaningless until you train/fine-tune the model.")

    return model


def _preprocess_for_crnn(crop: np.ndarray,
                         target_h: int = 32,
                         target_w: int = 128) -> "torch.Tensor":
    """Convert a BGR crop to a normalised (1, 1, H, W) float tensor."""
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_w, target_h))
    tensor  = torch.from_numpy(resized).float() / 255.0
    tensor  = tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    return tensor


def recognize_custom(crops: list[np.ndarray],
                     weights_path: str | None = None) -> list[str]:
    """
    Recognise text using the custom CRNN model.

    Args:
        crops:        List of BGR image crops.
        weights_path: Optional path to saved PyTorch state-dict (.pth file).

    Returns:
        List of recognised strings (will be mostly garbage without trained weights).
    """
    model = _load_custom_model(weights_path)
    if model is None:
        raise RuntimeError(
            "PyTorch is not installed.  Run:  pip install torch"
        )

    results = []
    with torch.no_grad():
        for i, crop in enumerate(crops):
            tensor   = _preprocess_for_crnn(crop)      # (1, 1, 32, 128)
            log_probs = model(tensor)                   # (T, 1, num_classes)
            text     = _ctc_decode(log_probs[:, 0, :]) # decode first (only) item
            results.append(text)
            print(f"[Recognition] Custom model crop {i+1}/{len(crops)}: '{text}'")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Auto-selecting public entry point
# ─────────────────────────────────────────────────────────────────────────────

def recognize(crops: list[np.ndarray],
              backend: str = "auto",
              languages: list[str] | None = None,
              tesseract_lang: str = "eng",
              custom_weights: str | None = None) -> list[str]:
    """
    Recognise text from a list of image crops.

    Args:
        crops:           List of BGR image crops (from detection.crop_regions).
        backend:         "auto"      – try EasyOCR → Tesseract → Custom.
                         "easyocr"   – force EasyOCR.
                         "tesseract" – force Tesseract.
                         "custom"    – force custom CRNN (needs trained weights).
        languages:       EasyOCR language list (e.g. ["en", "fr"]).
        tesseract_lang:  Tesseract language string (e.g. "eng+fra").
        custom_weights:  Path to custom model .pth weights file.

    Returns:
        List of recognised text strings, one per crop.
    """
    if not crops:
        print("[Recognition] No crops to process.")
        return []

    # ── Force specific backend ───────────────────────────────────────────────
    if backend == "easyocr":
        if not _EASYOCR_AVAILABLE:
            raise RuntimeError("easyocr is not installed.  pip install easyocr")
        return recognize_easyocr(crops, languages)

    if backend == "tesseract":
        return recognize_tesseract(crops, lang=tesseract_lang)

    if backend == "custom":
        return recognize_custom(crops, weights_path=custom_weights)

    # ── Auto: try in priority order ──────────────────────────────────────────
    if _EASYOCR_AVAILABLE:
        print("[Recognition] Using EasyOCR (auto-selected).")
        return recognize_easyocr(crops, languages)

    if _TESSERACT_AVAILABLE:
        print("[Recognition] EasyOCR not found – falling back to Tesseract.")
        return recognize_tesseract(crops, lang=tesseract_lang)

    if _TORCH_AVAILABLE:
        print("[Recognition] Tesseract not found – falling back to custom CRNN.")
        return recognize_custom(crops, weights_path=custom_weights)

    raise RuntimeError(
        "No OCR back-end is available.\n"
        "Install at least one of:\n"
        "  pip install easyocr\n"
        "  pip install pytesseract  (+ tesseract binary)\n"
        "  pip install torch"
    )
