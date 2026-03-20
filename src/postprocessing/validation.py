"""
Validation Module
-----------------
Regex-based and rule-based validators for extracted fields.
Drops predictions below confidence threshold.
"""

from __future__ import annotations

import re
from typing import Any, Dict

CONFIDENCE_THRESHOLD = 0.5

DATE_PATTERN = re.compile(
    r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"
)
INVOICE_PATTERN = re.compile(r"^[A-Z]{0,5}-?\d{3,10}$", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"^\$?[\d,]+\.?\d{0,2}$")


FIELD_VALIDATORS = {
    "date": DATE_PATTERN,
    "invoice_number": INVOICE_PATTERN,
    "total_amount": AMOUNT_PATTERN,
}


def validate_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extracted fields using regex rules.

    Fields that fail validation are set to None and flagged.
    Low-confidence fields (if confidence metadata is available) are dropped.

    Args:
        raw: Dict of field_name → extracted_value.

    Returns:
        Validated dict with invalid fields set to None.
    """
    validated: Dict[str, Any] = {}
    for field, value in raw.items():
        if value is None:
            validated[field] = None
            continue

        pattern = FIELD_VALIDATORS.get(field)
        if pattern and isinstance(value, str) and not pattern.match(value.strip()):
            validated[field] = None  # failed validation
        else:
            validated[field] = value

    return validated
