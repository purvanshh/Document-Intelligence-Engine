from __future__ import annotations

from evaluation.metrics import (
    compute_entity_metrics,
    compute_structured_output_metrics,
    compute_token_metrics,
    partial_match_score,
)


def test_token_metrics() -> None:
    metrics = compute_token_metrics(
        [["B-KEY", "I-KEY", "B-VALUE"]],
        [["B-KEY", "I-KEY", "B-VALUE"]],
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_entity_metrics() -> None:
    metrics = compute_entity_metrics(
        [[{"field": "invoice_number", "value": "INV-001"}]],
        [[{"field": "invoice_number", "value": "INV-001"}]],
    )
    assert metrics["f1"] == 1.0


def test_structured_output_metrics() -> None:
    metrics = compute_structured_output_metrics(
        {"invoice_number": {"value": "INV-001"}},
        {"invoice_number": {"value": "INV-001"}},
    )
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["field_level_accuracy"]["invoice_number"]["accuracy"] == 1.0


def test_partial_match_score() -> None:
    assert partial_match_score("INV-001", "INV-002") > 0.7
