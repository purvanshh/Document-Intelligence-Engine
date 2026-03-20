"""
Image Preprocessing Module
--------------------------
Normalize, denoise, and optionally correct skew in document images
before they are fed into the OCR pipeline.
"""

import cv2
import numpy as np


def normalize(image: np.ndarray, target_width: int = 1000) -> np.ndarray:
    """
    Resize image proportionally so the width equals `target_width`.

    Args:
        image: HxWx3 numpy array (RGB).
        target_width: Desired output width in pixels.

    Returns:
        Resized image as numpy array.
    """
    h, w = image.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized


def correct_skew(image: np.ndarray, delta: float = 1.0, limit: float = 5.0) -> np.ndarray:
    """
    Detect and correct document skew using Hough line transform.

    Args:
        image: HxWx3 numpy array.
        delta: Angle step for search (degrees).
        limit: Maximum rotation to consider (degrees).

    Returns:
        Deskewed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < limit:
                angles.append(angle)

    skew_angle = float(np.median(angles)) if angles else 0.0

    if abs(skew_angle) < 0.1:
        return image  # nothing to correct

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), skew_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return corrected


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
