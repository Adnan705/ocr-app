"""
utils.py
========
Helper functions for the OCR pipeline.

Covers:
  • Text post-processing (clean up raw OCR output)
  • Optional spell-check
  • Saving results to .txt files
  • Annotated image saving
  • Batch processing of multiple images
  • Simple table / form detection helper
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Text post-processing
# ─────────────────────────────────────────────────────────────────────────────

# Characters that are clearly OCR artefacts but not real punctuation
_JUNK_RE  = re.compile(r"[^\x20-\x7E\n]")     # non-printable ASCII
_MULTI_SP = re.compile(r" {2,}")               # 2+ consecutive spaces
_MULTI_NL = re.compile(r"\n{3,}")              # 3+ consecutive newlines


def clean_text(raw: str) -> str:
    """
    Remove common OCR artefacts and normalise whitespace.

    Steps
    -----
    1. Strip non-printable / non-ASCII characters.
    2. Collapse multiple spaces to one.
    3. Collapse 3+ newlines to 2 (preserve paragraph breaks).
    4. Strip leading / trailing whitespace from each line.
    5. Strip leading / trailing whitespace from the whole string.

    Args:
        raw: Raw string from the OCR engine.

    Returns:
        Cleaned string.
    """
    text = _JUNK_RE.sub("", raw)               # remove junk chars
    text = _MULTI_SP.sub(" ", text)            # collapse spaces
    text = _MULTI_NL.sub("\n\n", text)         # collapse blank lines

    # Strip each line individually
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)

    return text.strip()


def merge_texts(texts: list[str], separator: str = "\n") -> str:
    """
    Merge a list of per-region text strings into one document.

    Args:
        texts:     List of cleaned strings (one per detected text region).
        separator: String placed between consecutive regions.

    Returns:
        Single merged string.
    """
    non_empty = [t for t in texts if t.strip()]
    return separator.join(non_empty)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Optional spell-check (using pyspellchecker)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from spellchecker import SpellChecker as _SpellChecker
    _SPELLCHECK_AVAILABLE = True
except ImportError:
    _SPELLCHECK_AVAILABLE = False


def spell_check(text: str, language: str = "en") -> str:
    """
    Correct spelling mistakes in the OCR output.

    Requires: ``pip install pyspellchecker``

    Args:
        text:     Input text.
        language: Language code ("en", "es", "de", …).

    Returns:
        Text with misspelled words replaced by the most likely correction.
        If pyspellchecker is not installed, returns the original text unchanged.
    """
    if not _SPELLCHECK_AVAILABLE:
        print("[Utils] pyspellchecker not installed – skipping spell-check.  "
              "Run:  pip install pyspellchecker")
        return text

    spell  = _SpellChecker(language=language)
    words  = text.split()
    fixed  = []

    for word in words:
        # Strip punctuation before checking, re-attach afterwards
        stripped = word.strip(".,;:!?\"'()-")
        punct_l  = word[: len(word) - len(word.lstrip(".,;:!?\"'()-"))]
        punct_r  = word[len(word.rstrip(".,;:!?\"'()-")):]

        correction = spell.correction(stripped.lower())
        if correction and correction != stripped.lower():
            # Preserve original capitalisation style
            if stripped.isupper():
                correction = correction.upper()
            elif stripped[0].isupper():
                correction = correction.capitalize()
            fixed.append(punct_l + correction + punct_r)
        else:
            fixed.append(word)

    return " ".join(fixed)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_text(text: str, output_path: str) -> None:
    """
    Save extracted text to a plain-text file (UTF-8).

    Args:
        text:        Text to save.
        output_path: Destination file path (e.g. "output/result.txt").
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[Utils] Text saved → {output_path}")


def save_annotated_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an annotated (bounding-box overlay) image.

    Args:
        image:       BGR image array.
        output_path: Destination file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[Utils] Annotated image saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def get_image_paths(input_path: str,
                    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png",
                                                   ".bmp", ".tiff", ".webp")
                    ) -> list[str]:
    """
    Return a list of image file paths from either a single file or a directory.

    Args:
        input_path:  Path to a single image OR a directory containing images.
        extensions:  Accepted file extensions (lower-case).

    Returns:
        Sorted list of absolute image paths.
    """
    p = Path(input_path)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        paths = sorted(
            str(f) for f in p.iterdir()
            if f.suffix.lower() in extensions
        )
        if not paths:
            print(f"[Utils] No images found in '{input_path}'.")
        return paths

    raise FileNotFoundError(f"Path not found: {input_path}")


def batch_results_to_text(results: dict[str, str]) -> str:
    """
    Format a dict of {image_path: extracted_text} into a readable report.

    Args:
        results: Mapping of image path → recognised text.

    Returns:
        Formatted report string.
    """
    lines = []
    for path, text in results.items():
        lines.append("=" * 60)
        lines.append(f"File : {path}")
        lines.append("-" * 60)
        lines.append(text if text else "(no text detected)")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Simple table / form detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_table_lines(image_bgr: np.ndarray,
                       min_line_length: int = 100,
                       gap: int = 10) -> tuple[list, list]:
    """
    Detect horizontal and vertical lines that likely form a table or form.

    Uses morphological operations to isolate long lines.

    Args:
        image_bgr:       Input BGR image.
        min_line_length: Minimum line length in pixels to be considered.
        gap:             Structuring element gap parameter.

    Returns:
        Tuple (h_lines, v_lines) where each element is a list of bounding
        rectangles (x, y, w, h) for detected line segments.
    """
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Horizontal lines ─────────────────────────────────────────────────────
    h_kernel  = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_line_length, 1)
    )
    h_mask    = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_conts, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    h_lines   = [cv2.boundingRect(c) for c in h_conts]

    # ── Vertical lines ───────────────────────────────────────────────────────
    v_kernel  = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, min_line_length)
    )
    v_mask    = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_conts, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    v_lines   = [cv2.boundingRect(c) for c in v_conts]

    print(f"[Utils] Table detection: {len(h_lines)} horizontal, "
          f"{len(v_lines)} vertical line(s).")

    return h_lines, v_lines


def has_table(image_bgr: np.ndarray,
              min_h_lines: int = 2,
              min_v_lines: int = 2) -> bool:
    """
    Heuristic check: does the image contain a table?

    Args:
        image_bgr:   Input BGR image.
        min_h_lines: Minimum horizontal lines required.
        min_v_lines: Minimum vertical lines required.

    Returns:
        True if the image likely contains a table.
    """
    h, v = detect_table_lines(image_bgr)
    return len(h) >= min_h_lines and len(v) >= min_v_lines


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(text: str, width: int = 80) -> None:
    """
    Print the final extracted text to stdout with a decorative border.

    Args:
        text:  The recognised text.
        width: Console width for the border.
    """
    border = "─" * width
    print(f"\n{'═' * width}")
    print(" EXTRACTED TEXT ".center(width, "═"))
    print(f"{'═' * width}\n")
    # Word-wrap long lines for readability
    for para in text.split("\n"):
        if para.strip():
            wrapped = textwrap.fill(para, width=width)
            print(wrapped)
        else:
            print()
    print(f"\n{border}\n")
