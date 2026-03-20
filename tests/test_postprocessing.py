from __future__ import annotations

from datetime import date, timedelta

from postprocessing.constraints import apply_constraints
from postprocessing.entity_grouping import group_entities
from postprocessing.normalization import normalize_currency, normalize_date
from postprocessing.pipeline import postprocess_predictions
from postprocessing.validation import validate_fields

from document_intelligence_engine.core.config import get_settings


def test_entity_grouping_correctness() -> None:
    grouped, errors = group_entities(
        [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.95},
            {"text": "Number", "label": "I-KEY", "confidence": 0.94},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.91},
        ],
        get_settings().postprocessing.field_aliases,
    )

    assert errors == []
    assert grouped == [
        {
            "key": "Invoice Number",
            "field": "invoice_number",
            "value": "INV-1023",
            "confidence": 0.9275,
            "key_confidence": 0.945,
            "value_confidence": 0.91,
        }
    ]


def test_normalization_accuracy() -> None:
    assert normalize_date("12/01/2025") == "2025-01-12"
    assert normalize_currency("$1,200.50") == 1200.5


def test_validation_rules() -> None:
    settings = get_settings()
    document, errors = validate_fields(
        [
            {
                "key": "Invoice Number",
                "field": "invoice_number",
                "value": "INV-1023",
                "confidence": 0.91,
            },
            {
                "key": "Date",
                "field": "date",
                "value": "2025-01-12",
                "confidence": 0.90,
            },
            {
                "key": "Total Amount",
                "field": "total_amount",
                "value": 1200.5,
                "confidence": 0.88,
            },
        ],
        settings,
    )

    assert errors == []
    assert document["invoice_number"]["valid"] is True
    assert document["date"]["valid"] is True
    assert document["total_amount"]["valid"] is True


def test_constraint_logic() -> None:
    settings = get_settings()
    future_date = (date.today() + timedelta(days=1)).isoformat()
    document, errors, flags = apply_constraints(
        {
            "invoice_number": {"value": "INV-1023", "confidence": 0.9, "valid": True},
            "date": {"value": future_date, "confidence": 0.9, "valid": True},
            "total_amount": {"value": 100.0, "confidence": 0.9, "valid": True},
            "line_items": {
                "value": [{"price": 25.0, "quantity": 2}, {"price": 10.0, "quantity": 1}],
                "confidence": 0.9,
                "valid": True,
            },
        },
        settings,
    )

    assert document["date"]["valid"] is False
    assert document["total_amount"]["valid"] is False
    assert "future_date:date" in flags
    assert "line_items_total_mismatch" in flags
    assert len(errors) == 2


def test_postprocess_pipeline_output() -> None:
    output = postprocess_predictions(
        [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.95},
            {"text": "Number", "label": "I-KEY", "confidence": 0.94},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.91},
            {"text": "Date", "label": "B-KEY", "confidence": 0.97},
            {"text": "12/01/2025", "label": "B-VALUE", "confidence": 0.93},
            {"text": "Total", "label": "B-KEY", "confidence": 0.96},
            {"text": "Amount", "label": "I-KEY", "confidence": 0.95},
            {"text": "$1,200.50", "label": "B-VALUE", "confidence": 0.92},
        ]
    )

    assert output["invoice_number"]["value"] == "INV-1023"
    assert output["invoice_number"]["valid"] is True
    assert output["date"]["value"] == "2025-01-12"
    assert output["total_amount"]["value"] == 1200.5
    assert output["_errors"] == []
    assert output["_constraint_flags"] == []
