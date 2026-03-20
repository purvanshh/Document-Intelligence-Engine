"""
Bounding Box Alignment Module
------------------------------
Normalises OCR bounding boxes to the coordinate space expected by
LayoutLMv3 (0–1000 range), and utilities for overlap / sorting.
"""

from typing import List, Tuple


def normalize_bbox(
    bbox: List[int], image_width: int, image_height: int, scale: int = 1000
) -> List[int]:
    """
    Normalise an absolute-pixel bounding box to [0, scale] range.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in pixels.
        image_width:  Original image width.
        image_height: Original image height.
        scale: Target normalisation range (1000 for LayoutLMv3).

    Returns:
        Normalised bbox list.
    """
    x_min, y_min, x_max, y_max = bbox
    return [
        int(x_min / image_width * scale),
        int(y_min / image_height * scale),
        int(x_max / image_width * scale),
        int(y_max / image_height * scale),
    ]


def sort_reading_order(
    tokens: List[dict], line_threshold: int = 10
) -> List[dict]:
    """
    Sort tokens in natural reading order (top-to-bottom, left-to-right).

    Args:
        tokens: List of dicts with at least a 'bbox' key.
        line_threshold: Vertical pixel distance to consider as the same line.

    Returns:
        Sorted list of tokens.
    """
    return sorted(tokens, key=lambda t: (t["bbox"][1] // line_threshold, t["bbox"][0]))


def iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute Intersection over Union for two bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)
