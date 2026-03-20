"""
Normalization Module
--------------------
Convert extracted raw strings into canonical typed forms.
E.g., "01/12/2025" → "2025-01-12", "$1,200.50" → 1200.50
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Date normalisation ──────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%d %b %Y", "%B %d, %Y",
]


def parse_date(raw: str) -> Optional[str]:
    """Try to parse `raw` as a date; return ISO 8601 string or None."""
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ── Currency normalisation ──────────────────────────────────────────────────

def parse_amount(raw: str) -> Optional[float]:
    """Strip currency symbols/commas and return a float, or None."""
    raw = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        return float(raw)
    except ValueError:
        return None


# ── OCR typo correction ─────────────────────────────────────────────────────

_TYPO_MAP = {
    "O": "0",   # capital O → zero  (inside numeric tokens only)
    "l": "1",   # lowercase l → one
    "S": "5",   # S → 5 in numbers
    "I": "1",
}

def correct_ocr_typos(token: str) -> str:
    """
    Apply common OCR digit-substitution corrections.
    Only applied if the token looks mostly numeric.
    """
    if not re.search(r"\d", token):
        return token  # doesn't look numeric, leave it alone
    for wrong, right in _TYPO_MAP.items():
        token = token.replace(wrong, right)
    return token


# ── Field-level dispatcher ──────────────────────────────────────────────────

def normalize_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply type-specific normalisation to known field names.

    Args:
        data: Dict from the validation step.

    Returns:
        New dict with normalised values.
    """
    result = dict(data)

    # Date fields
    for key in ("date", "invoice_date", "due_date"):
        if isinstance(result.get(key), str):
            result[key] = parse_date(result[key]) or result[key]

    # Amount fields
    for key in ("total_amount", "subtotal", "tax", "discount"):
        if isinstance(result.get(key), str):
            result[key] = parse_amount(result[key])

    return result
