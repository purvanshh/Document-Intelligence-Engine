"""Evaluation metrics."""

from __future__ import annotations


def compute_exact_match(prediction: dict[str, object], ground_truth: dict[str, object]) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def compute_field_level_accuracy(
    predictions: list[dict[str, object]],
    ground_truths: list[dict[str, object]],
    fields: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not predictions or not ground_truths:
        return {field: 0.0 for field in fields}

    sample_count = min(len(predictions), len(ground_truths))
    for field in fields:
        matches = 0
        for index in range(sample_count):
            if predictions[index].get(field) == ground_truths[index].get(field):
                matches += 1
        results[field] = matches / sample_count
    return results
