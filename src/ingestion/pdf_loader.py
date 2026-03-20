"""
PDF Loader Module
-----------------
Handles ingestion of PDF files and conversion to image tensors
for downstream processing.
"""

import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


def load_pdf(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """
    Load a PDF file and convert each page to a numpy image array.

    Args:
        pdf_path: Absolute or relative path to PDF file.
        dpi: Resolution for rendering. Higher = more detail, slower.

    Returns:
        List of numpy arrays (H, W, 3) in RGB channel order, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: List[np.ndarray] = []

    zoom = dpi / 72  # default PDF DPI is 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(np.array(img))

    doc.close()
    return pages


def is_supported_file(file_path: str) -> bool:
    """Return True if the file extension is supported (PDF / image)."""
    supported = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    return Path(file_path).suffix.lower() in supported
