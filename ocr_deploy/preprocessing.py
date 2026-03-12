"""
preprocessing.py
================
Image cleaning and preprocessing functions for the OCR pipeline.

Steps performed:
  1. Read the input image
  2. Convert to grayscale
  3. Denoise using Non-Local Means or Gaussian blur
  4. Binarize with Otsu's or adaptive thresholding
  5. Deskew / correct rotation
  6. Optionally resize for model input
"""

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1.  Load image
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR NumPy array (OpenCV format).

    Args:
        image_path: Path to the input image (.jpg / .png / etc.).

    Returns:
        BGR image as a NumPy uint8 array.

    Raises:
        FileNotFoundError: If the path does not exist or cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    print(f"[Preprocessing] Loaded image: {image_path}  shape={img.shape}")
    return img


# ---------------------------------------------------------------------------
# 2.  Grayscale conversion
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(img.shape) == 2:
        # Already grayscale
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[Preprocessing] Converted to grayscale.")
    return gray


# ---------------------------------------------------------------------------
# 3.  Denoising
# ---------------------------------------------------------------------------

def denoise(gray: np.ndarray, method: str = "nlm") -> np.ndarray:
    """
    Remove noise from a grayscale image.

    Args:
        gray:   Grayscale image.
        method: "nlm"      – Non-Local Means (better quality, slower)
                "gaussian" – Gaussian blur   (fast, good for mild noise)
                "median"   – Median filter   (good for salt-and-pepper noise)

    Returns:
        Denoised grayscale image.
    """
    if method == "nlm":
        denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                            templateWindowSize=7,
                                            searchWindowSize=21)
    elif method == "gaussian":
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    elif method == "median":
        denoised = cv2.medianBlur(gray, 3)
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    print(f"[Preprocessing] Denoised with method='{method}'.")
    return denoised


# ---------------------------------------------------------------------------
# 4.  Binarisation / Thresholding
# ---------------------------------------------------------------------------

def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Convert grayscale image to a binary (black/white) image.

    Args:
        gray:   Grayscale image.
        method: "otsu"     – Otsu's global threshold (best for uniform lighting)
                "adaptive" – Adaptive mean threshold (best for varying lighting)
                "simple"   – Fixed threshold at 127

    Returns:
        Binary image (0 = background, 255 = foreground/text).
    """
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       blockSize=15, C=10)
    elif method == "simple":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    print(f"[Preprocessing] Binarized with method='{method}'.")
    return binary


# ---------------------------------------------------------------------------
# 5.  Deskewing
# ---------------------------------------------------------------------------

def _compute_skew_angle(binary: np.ndarray) -> float:
    """
    Estimate the skew angle of text in a binary image using the Hough
    line transform.

    Returns the detected angle in degrees (positive = clockwise tilt).
    """
    # Invert so text pixels are white on black background
    inverted = cv2.bitwise_not(binary)

    # Find coordinates of all white (text) pixels
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 5:
        return 0.0

    # minAreaRect returns (center, (w, h), angle)
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle to the range (-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    return angle


def deskew(binary: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Correct slight rotational skew in a binary image.

    Args:
        binary:    Binary image to deskew.
        max_angle: Only correct if detected angle < max_angle degrees
                   (avoids misidentifying intentional rotations).

    Returns:
        Deskewed binary image.
    """
    angle = _compute_skew_angle(binary)
    print(f"[Preprocessing] Detected skew angle: {angle:.2f}°")

    if abs(angle) < 0.5 or abs(angle) > max_angle:
        print("[Preprocessing] Skew angle negligible or too large – skipping correction.")
        return binary

    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    deskewed = cv2.warpAffine(binary, rotation_matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    print(f"[Preprocessing] Deskewed by {angle:.2f}°.")
    return deskewed


# ---------------------------------------------------------------------------
# 6.  Resize
# ---------------------------------------------------------------------------

def resize_for_model(img: np.ndarray,
                     target_height: int = 640,
                     max_width: int = 1280) -> np.ndarray:
    """
    Resize image so its height equals `target_height` while preserving
    the aspect ratio.  Width is capped at `max_width`.

    Args:
        img:           Input image (grayscale or colour).
        target_height: Desired output height in pixels.
        max_width:     Maximum allowed width.

    Returns:
        Resized image.
    """
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = min(int(w * scale), max_width)
    resized = cv2.resize(img, (new_w, target_height),
                         interpolation=cv2.INTER_AREA)
    print(f"[Preprocessing] Resized from ({h}, {w}) → ({target_height}, {new_w}).")
    return resized


# ---------------------------------------------------------------------------
# 7.  Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(image_path: str,
               denoise_method: str = "nlm",
               binarize_method: str = "otsu",
               do_deskew: bool = True,
               do_resize: bool = False,
               target_height: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the complete preprocessing pipeline on a single image.

    Args:
        image_path:      Path to the input image.
        denoise_method:  See denoise().
        binarize_method: See binarize().
        do_deskew:       Whether to apply deskewing.
        do_resize:       Whether to resize for model input.
        target_height:   Target height when do_resize=True.

    Returns:
        Tuple (original_bgr, preprocessed_binary):
            original_bgr       – Original colour image (for detection overlay).
            preprocessed_binary – Final cleaned binary image.
    """
    original = load_image(image_path)
    gray      = to_grayscale(original)
    denoised  = denoise(gray, method=denoise_method)
    binary    = binarize(denoised, method=binarize_method)

    if do_deskew:
        binary = deskew(binary)

    if do_resize:
        binary = resize_for_model(binary, target_height=target_height)

    print("[Preprocessing] Pipeline complete.\n")
    return original, binary
