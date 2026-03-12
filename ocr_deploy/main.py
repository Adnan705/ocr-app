"""
main.py
=======
Entry point for the open-source Image-to-Text OCR pipeline.

Usage
-----
Single image:
    python main.py --input path/to/image.jpg

Directory (batch mode):
    python main.py --input path/to/folder/

Full options:
    python main.py --input image.jpg \\
                   --output results/output.txt \\
                   --detect_method auto \\
                   --ocr_backend auto \\
                   --languages en \\
                   --denoise nlm \\
                   --binarize otsu \\
                   --no_deskew \\
                   --spell_check \\
                   --save_annotated \\
                   --detect_table

Pipeline
--------
  1. Preprocessing  – load → grayscale → denoise → binarize → deskew
  2. Detection      – find text regions (EAST or contour-based)
  3. Recognition    – OCR each region  (EasyOCR / Tesseract / custom CRNN)
  4. Postprocessing – clean text, optional spell-check
  5. Output         – print to screen + save .txt + optional annotated image
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Project modules ──────────────────────────────────────────────────────────
from preprocessing import preprocess
from detection     import detect
from recognition   import recognize
from utils         import (
    clean_text,
    merge_texts,
    spell_check,
    save_text,
    save_annotated_image,
    get_image_paths,
    batch_results_to_text,
    has_table,
    print_results,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline (single image)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path:      str,
    output_txt:      str  | None = None,
    output_img:      str  | None = None,
    detect_method:   str         = "auto",
    ocr_backend:     str         = "auto",
    languages:       list[str]   | None = None,
    denoise_method:  str         = "nlm",
    binarize_method: str         = "otsu",
    do_deskew:       bool        = True,
    do_spell_check:  bool        = False,
    detect_table:    bool        = False,
    custom_weights:  str  | None = None,
) -> str:
    """
    Run the complete OCR pipeline on a single image.

    Args:
        image_path:      Path to the input image.
        output_txt:      Where to save the extracted text (.txt).
                         Defaults to  <input_stem>_extracted.txt  in ./output/.
        output_img:      Where to save the annotated image.
                         Defaults to  <input_stem>_annotated.jpg  in ./output/.
        detect_method:   "auto" | "east" | "contour"
        ocr_backend:     "auto" | "easyocr" | "tesseract" | "custom"
        languages:       EasyOCR language list  (e.g. ["en", "fr"]).
        denoise_method:  "nlm" | "gaussian" | "median"
        binarize_method: "otsu" | "adaptive" | "simple"
        do_deskew:       Whether to apply deskew correction.
        do_spell_check:  Whether to run spell-checker post-processing.
        detect_table:    Print a note if a table/form is detected.
        custom_weights:  Path to custom CRNN weights (.pth).

    Returns:
        Extracted (and cleaned) text string.
    """
    stem = Path(image_path).stem

    # Default output paths
    os.makedirs("output", exist_ok=True)
    if output_txt is None:
        output_txt = f"output/{stem}_extracted.txt"
    if output_img is None:
        output_img = f"output/{stem}_annotated.jpg"

    print(f"\n{'=' * 60}")
    print(f" Processing: {image_path}")
    print(f"{'=' * 60}\n")
    t0 = time.time()

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    print("── Step 1: Preprocessing ──────────────────────────────────")
    original_bgr, _ = preprocess(
        image_path,
        denoise_method=denoise_method,
        binarize_method=binarize_method,
        do_deskew=do_deskew,
    )

    # ── Step 2: Text detection ───────────────────────────────────────────────
    print("── Step 2: Text Detection ─────────────────────────────────")
    boxes, crops, annotated = detect(original_bgr, method=detect_method)

    if detect_table:
        if has_table(original_bgr):
            print("[Main] ⚠  Table / form structure detected in this image.")

    if not crops:
        print("[Main] No text regions detected – output will be empty.")
        save_text("", output_txt)
        save_annotated_image(annotated, output_img)
        return ""

    # ── Step 3: Text recognition ─────────────────────────────────────────────
    print("── Step 3: Text Recognition ───────────────────────────────")
    raw_texts = recognize(
        crops,
        backend=ocr_backend,
        languages=languages,
        custom_weights=custom_weights,
    )

    # ── Step 4: Postprocessing ───────────────────────────────────────────────
    print("── Step 4: Postprocessing ─────────────────────────────────")
    cleaned_texts = [clean_text(t) for t in raw_texts]
    merged        = merge_texts(cleaned_texts)

    if do_spell_check:
        lang_code = (languages[0] if languages else "en")
        merged    = spell_check(merged, language=lang_code)
        print(f"[Main] Spell-check applied (language='{lang_code}').")

    # ── Step 5: Output ───────────────────────────────────────────────────────
    print("── Step 5: Output ─────────────────────────────────────────")
    print_results(merged)
    save_text(merged, output_txt)
    save_annotated_image(annotated, output_img)

    elapsed = time.time() - t0
    print(f"[Main] ✓ Done in {elapsed:.2f}s")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Batch pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    input_path:      str,
    output_dir:      str         = "output",
    detect_method:   str         = "auto",
    ocr_backend:     str         = "auto",
    languages:       list[str]   | None = None,
    denoise_method:  str         = "nlm",
    binarize_method: str         = "otsu",
    do_deskew:       bool        = True,
    do_spell_check:  bool        = False,
    detect_table:    bool        = False,
    custom_weights:  str  | None = None,
) -> dict[str, str]:
    """
    Run the OCR pipeline on every image in a directory (or a single file).

    Returns:
        Dict mapping each image path to its extracted text.
    """
    image_paths = get_image_paths(input_path)
    if not image_paths:
        print("[Batch] No images to process.")
        return {}

    print(f"[Batch] Found {len(image_paths)} image(s) to process.")
    results = {}

    for i, path in enumerate(image_paths, start=1):
        print(f"\n[Batch] Image {i}/{len(image_paths)}: {path}")
        stem       = Path(path).stem
        output_txt = os.path.join(output_dir, f"{stem}_extracted.txt")
        output_img = os.path.join(output_dir, f"{stem}_annotated.jpg")

        try:
            text = run_pipeline(
                image_path      = path,
                output_txt      = output_txt,
                output_img      = output_img,
                detect_method   = detect_method,
                ocr_backend     = ocr_backend,
                languages       = languages,
                denoise_method  = denoise_method,
                binarize_method = binarize_method,
                do_deskew       = do_deskew,
                do_spell_check  = do_spell_check,
                detect_table    = detect_table,
                custom_weights  = custom_weights,
            )
            results[path] = text
        except Exception as exc:
            print(f"[Batch] ✗ Error processing {path}: {exc}")
            results[path] = ""

    # Save combined batch report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "batch_report.txt")
    report_text = batch_results_to_text(results)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n[Batch] Combined report saved → {report_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Open-source Image-to-Text OCR Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--input", "-i", required=True,
        help="Path to an image file (jpg/png) or a directory of images.",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output .txt path (single-image mode).  Default: output/<stem>_extracted.txt",
    )
    p.add_argument(
        "--output_dir", default="output",
        help="Output directory (batch mode).  Default: output/",
    )

    # Detection
    p.add_argument(
        "--detect_method", default="auto",
        choices=["auto", "east", "contour"],
        help="Text detection method (default: auto).",
    )

    # Recognition
    p.add_argument(
        "--ocr_backend", default="auto",
        choices=["auto", "easyocr", "tesseract", "custom"],
        help="OCR recognition backend (default: auto).",
    )
    p.add_argument(
        "--languages", "-l", default="en",
        help="Comma-separated EasyOCR language codes  e.g. 'en,fr' (default: en).",
    )
    p.add_argument(
        "--custom_weights", default=None,
        help="Path to custom CRNN model weights (.pth).",
    )

    # Preprocessing
    p.add_argument(
        "--denoise", default="nlm",
        choices=["nlm", "gaussian", "median"],
        help="Denoising method (default: nlm).",
    )
    p.add_argument(
        "--binarize", default="otsu",
        choices=["otsu", "adaptive", "simple"],
        help="Binarization method (default: otsu).",
    )
    p.add_argument(
        "--no_deskew", action="store_true",
        help="Disable skew correction.",
    )

    # Enhancements
    p.add_argument(
        "--spell_check", action="store_true",
        help="Apply spell-checker post-processing (requires pyspellchecker).",
    )
    p.add_argument(
        "--save_annotated", action="store_true",
        help="Save image with bounding boxes drawn (always saved by default).",
    )
    p.add_argument(
        "--detect_table", action="store_true",
        help="Print a note if a table / form is detected in the image.",
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]

    input_p = Path(args.input)

    if input_p.is_dir():
        # ── Batch mode ───────────────────────────────────────────────────────
        run_batch(
            input_path      = str(input_p),
            output_dir      = args.output_dir,
            detect_method   = args.detect_method,
            ocr_backend     = args.ocr_backend,
            languages       = languages,
            denoise_method  = args.denoise,
            binarize_method = args.binarize,
            do_deskew       = not args.no_deskew,
            do_spell_check  = args.spell_check,
            detect_table    = args.detect_table,
            custom_weights  = args.custom_weights,
        )
    else:
        # ── Single image mode ─────────────────────────────────────────────────
        run_pipeline(
            image_path      = str(input_p),
            output_txt      = args.output,
            detect_method   = args.detect_method,
            ocr_backend     = args.ocr_backend,
            languages       = languages,
            denoise_method  = args.denoise,
            binarize_method = args.binarize,
            do_deskew       = not args.no_deskew,
            do_spell_check  = args.spell_check,
            detect_table    = args.detect_table,
            custom_weights  = args.custom_weights,
        )


if __name__ == "__main__":
    main()
