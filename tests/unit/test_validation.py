from __future__ import annotations

from postprocessing.confidence import apply_confidence_policy
from postprocessing.constraints import apply_constraints
from postprocessing.validation import validate_fields


def test_validation_rules(settings):
    document, errors = validate_fields(
        [
            {"field": "invoice_number", "value": "INV-1023", "confidence": 0.91, "key": "Invoice Number"},
            {"field": "date", "value": "2025-01-12", "confidence": 0.92, "key": "Date"},
            {"field": "total_amount", "value": 1200.5, "confidence": 0.93, "key": "Total Amount"},
        ],
        settings,
    )
    assert errors == []
    assert document["invoice_number"]["valid"] is True


def test_constraint_logic(settings):
    document, errors, flags = apply_constraints(
        {
            "invoice_number": {"value": "INV-1023", "confidence": 0.9, "valid": True},
            "date": {"value": "2999-01-01", "confidence": 0.9, "valid": True},
            "total_amount": {"value": 10.0, "confidence": 0.9, "valid": True},
            "line_items": {"value": [{"price": 3.0, "quantity": 2}], "confidence": 0.9, "valid": True},
        },
        settings,
    )
    assert errors
    assert "future_date:date" in flags
    assert "line_items_total_mismatch" in flags


def test_confidence_threshold_policy(settings):
    filtered, errors = apply_confidence_policy(
        {
            "invoice_number": {"value": "INV-1023", "confidence": 0.95, "valid": True},
            "total_amount": {"value": 1200.5, "confidence": 0.2, "valid": True},
        },
        settings,
    )
    assert "invoice_number" in filtered
    assert "total_amount" not in filtered
    assert errors
