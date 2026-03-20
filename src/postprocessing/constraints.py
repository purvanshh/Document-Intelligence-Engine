"""
Constraints Module
------------------
Cross-field consistency checks.
E.g., sum(line_items) ≈ total_amount
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


def apply_constraints(data: Dict[str, Any], tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Apply cross-field business rules to the extracted document data.

    Currently implemented rules:
        1. line_items_sum ≈ total_amount  (within `tolerance` fraction)

    Args:
        data: Normalized extraction dict.
        tolerance: Fractional difference allowed (default 5%).

    Returns:
        Data augmented with a '_constraint_flags' key listing violated rules.
    """
    flags: List[str] = []

    # Rule 1: line item sum vs total
    line_items: List[Dict] = data.get("line_items", [])
    total: Optional[float] = data.get("total_amount")

    if line_items and total is not None:
        computed_sum = _sum_line_items(line_items)
        if computed_sum is not None:
            diff = abs(computed_sum - total)
            if diff > tolerance * max(abs(total), 1e-6):
                flags.append(
                    f"line_items_sum_mismatch: computed={computed_sum:.2f}, reported={total:.2f}"
                )

    data["_constraint_flags"] = flags
    return data


def _sum_line_items(items: List[Dict]) -> Optional[float]:
    """Sum price * quantity for each line item; return None on parse error."""
    total = 0.0
    for item in items:
        try:
            price = float(item.get("price", 0))
            qty = float(item.get("quantity", 1))
            total += price * qty
        except (TypeError, ValueError):
            return None
    return total
