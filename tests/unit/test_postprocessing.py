from __future__ import annotations

from postprocessing.entity_grouping import group_entities
from postprocessing.normalization import normalize_currency, normalize_date
from postprocessing.pipeline import postprocess_predictions


def test_entity_grouping_correctness(settings, mock_model_output):
    grouped, errors = group_entities(mock_model_output[:3], settings.postprocessing.field_aliases)
    assert errors == []
    assert grouped[0]["key"] == "Invoice Number"
    assert grouped[0]["value"] == "INV-1023"


def test_normalization_accuracy():
    assert normalize_date("12/01/2025") == "2025-01-12"
    assert normalize_currency("$1,200.50") == 1200.5


def test_postprocessing_output_schema(mock_model_output):
    output = postprocess_predictions(mock_model_output)
    assert output["invoice_number"]["value"] == "INV-1023"
    assert output["total_amount"]["value"] == 1200.5
    assert "_errors" in output
