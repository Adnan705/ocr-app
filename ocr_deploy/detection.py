"""
detection.py
============
Text-region detection for the OCR pipeline.

Two detection strategies are provided:

  A) EAST detector  – Deep-learning based, very accurate for scene text.
                      Requires the pre-trained EAST model file
                      (frozen_east_text_detection.pb).

  B) Contour-based  – Pure OpenCV, zero external model files needed.
                      Works well for document / printed text.

The pipeline automatically falls back to the contour method if the EAST
model file is not present.
"""

import os
import urllib.request

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EAST_MODEL_URL = (
    "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/"
    "master/frozen_east_text_detection.pb"
)
EAST_MODEL_PATH = "models/frozen_east_text_detection.pb"

# EAST input dimensions must be multiples of 32
EAST_INPUT_W = 320
EAST_INPUT_H = 320

# Confidence threshold for EAST detections
EAST_SCORE_THRESH = 0.5
EAST_NMS_THRESH   = 0.4


# ---------------------------------------------------------------------------
# EAST helpers
# ---------------------------------------------------------------------------

def _download_east_model() -> bool:
    """
    Try to download the EAST model weights from GitHub.

    Returns True on success, False on failure.
    """
    os.makedirs("models", exist_ok=True)
    print(f"[Detection] Downloading EAST model from:\n  {EAST_MODEL_URL}")
    try:
        urllib.request.urlretrieve(EAST_MODEL_URL, EAST_MODEL_PATH)
        print("[Detection] EAST model downloaded successfully.")
        return True
    except Exception as exc:
        print(f"[Detection] Download failed: {exc}")
        return False


def _load_east_model():
    """Load the EAST model; download it first if missing."""
    if not os.path.exists(EAST_MODEL_PATH):
        success = _download_east_model()
        if not success:
            return None
    try:
        net = cv2.dnn.readNet(EAST_MODEL_PATH)
        print("[Detection] EAST model loaded.")
        return net
    except Exception as exc:
        print(f"[Detection] Could not load EAST model: {exc}")
        return None


def _east_decode_predictions(scores, geometry):
    """
    Decode the EAST output layers into bounding rectangles and confidences.

    Args:
        scores:   Score map output from EAST  (1, 1, H, W).
        geometry: Geometry map output from EAST (1, 5, H, W).

    Returns:
        rects:         List of (x, y, w, h) rectangles.
        confidences:   Corresponding confidence scores.
    """
    (numRows, numCols) = scores.shape[2:4]
    rects       = []
    confidences = []

    for y in range(numRows):
        scoresData   = scores[0, 0, y]
        xData0       = geometry[0, 0, y]
        xData1       = geometry[0, 1, y]
        xData2       = geometry[0, 2, y]
        xData3       = geometry[0, 3, y]
        anglesData   = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < EAST_SCORE_THRESH:
                continue

            # Compute the offset factor (each cell maps to 4 px)
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Rotation angle
            angle  = anglesData[x]
            cos_a  = np.cos(angle)
            sin_a  = np.sin(angle)

            # Box dimensions
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Rotated bounding box center
            endX   = int(offsetX + (cos_a * xData1[x]) + (sin_a * xData2[x]))
            endY   = int(offsetY - (sin_a * xData1[x]) + (cos_a * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    return rects, confidences


def detect_east(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect text regions using the EAST deep-learning model.

    Args:
        image_bgr: Original BGR image.

    Returns:
        List of bounding boxes as (x, y, w, h) in original-image coordinates.
    """
    net = _load_east_model()
    if net is None:
        print("[Detection] Falling back to contour detection.")
        return detect_contours(image_bgr)

    orig_h, orig_w = image_bgr.shape[:2]

    # Resize to EAST input size
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0,
        size=(EAST_INPUT_W, EAST_INPUT_H),
        mean=(123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)

    # Forward pass – get score map and geometry map
    layer_names = ["feature_fusion/Conv_7/Sigmoid",
                   "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layer_names)

    # Decode raw predictions
    rects, confidences = _east_decode_predictions(scores, geometry)
    if not rects:
        print("[Detection] EAST found no text regions.")
        return []

    # Convert (x, y, w, h) → (x1, y1, x2, y2) for NMS
    boxes_xyxy = [(r[0], r[1], r[0] + r[2], r[1] + r[3]) for r in rects]

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        [(r[0], r[1], r[2], r[3]) for r in rects],
        confidences,
        EAST_SCORE_THRESH,
        EAST_NMS_THRESH,
    )

    # Scale boxes back to original image dimensions
    rW = orig_w / EAST_INPUT_W
    rH = orig_h / EAST_INPUT_H

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = rects[i]
            x1 = max(0, int(x * rW))
            y1 = max(0, int(y * rH))
            x2 = min(orig_w, int((x + w) * rW))
            y2 = min(orig_h, int((y + h) * rH))
            final_boxes.append((x1, y1, x2 - x1, y2 - y1))

    print(f"[Detection] EAST detected {len(final_boxes)} text region(s).")
    return final_boxes


# ---------------------------------------------------------------------------
# Contour-based detection (no model needed)
# ---------------------------------------------------------------------------

def detect_contours(image_bgr: np.ndarray,
                    min_area: int = 100,
                    padding: int = 4) -> list[tuple[int, int, int, int]]:
    """
    Detect text regions using morphological operations + contour analysis.

    This method works best for documents with clear text on plain backgrounds.

    Args:
        image_bgr: Original BGR image.
        min_area:  Minimum contour area (px²) to keep.
        padding:   Extra pixels to pad each bounding box.

    Returns:
        List of bounding boxes as (x, y, w, h).
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge letters into word blobs
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(dilated,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image_bgr.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        # Add padding, clamped to image bounds
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w_img - x, w + 2 * padding)
        h = min(h_img - y, h + 2 * padding)
        boxes.append((x, y, w, h))

    print(f"[Detection] Contour method detected {len(boxes)} text region(s).")
    return boxes


# ---------------------------------------------------------------------------
# Visualise detections
# ---------------------------------------------------------------------------

def draw_bounding_boxes(image_bgr: np.ndarray,
                        boxes: list[tuple[int, int, int, int]],
                        color: tuple = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on a copy of the image.

    Args:
        image_bgr:  Original BGR image.
        boxes:      List of (x, y, w, h) bounding boxes.
        color:      Box colour in BGR.
        thickness:  Line thickness in pixels.

    Returns:
        Annotated image copy.
    """
    annotated = image_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


# ---------------------------------------------------------------------------
# Crop text regions
# ---------------------------------------------------------------------------

def crop_regions(image_bgr: np.ndarray,
                 boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    """
    Crop each bounding box from the image and return as a list of patches.

    Boxes are sorted top-to-bottom then left-to-right (reading order).

    Args:
        image_bgr: Original BGR image.
        boxes:     List of (x, y, w, h) bounding boxes.

    Returns:
        List of BGR image crops.
    """
    # Sort: primary key = y (top), secondary = x (left)
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    crops = []
    for (x, y, w, h) in sorted_boxes:
        crop = image_bgr[y:y + h, x:x + w]
        if crop.size > 0:
            crops.append(crop)

    print(f"[Detection] Cropped {len(crops)} region(s).")
    return crops


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect(image_bgr: np.ndarray,
           method: str = "auto") -> tuple[list, list, np.ndarray]:
    """
    Run text detection on a BGR image.

    Args:
        image_bgr: Original BGR image.
        method:    "east"     – Use EAST model (downloads if missing).
                   "contour"  – Use contour-based method.
                   "auto"     – Try EAST; fall back to contours.

    Returns:
        Tuple (boxes, crops, annotated_image):
            boxes           – List of (x, y, w, h) bounding boxes.
            crops           – Cropped text patches in reading order.
            annotated_image – Original image with boxes drawn.
    """
    if method == "east":
        boxes = detect_east(image_bgr)
    elif method == "contour":
        boxes = detect_contours(image_bgr)
    else:  # "auto"
        if os.path.exists(EAST_MODEL_PATH):
            boxes = detect_east(image_bgr)
        else:
            print("[Detection] EAST model not found – using contour detection.")
            boxes = detect_contours(image_bgr)

    annotated = draw_bounding_boxes(image_bgr, boxes)
    crops     = crop_regions(image_bgr, boxes)

    return boxes, crops, annotated
