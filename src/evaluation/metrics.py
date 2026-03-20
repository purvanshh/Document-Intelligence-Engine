"""
Evaluation Metrics Module
-------------------------
Precision, Recall, F1, and Exact Match at the field level.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def exact_match(pred: Any, gold: Any) -> bool:
    """Return True if prediction exactly matches ground-truth (after normalisation)."""
    return str(pred).strip().lower() == str(gold).strip().lower()


def compute_field_f1(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-field Precision, Recall, F1, and Exact Match across a dataset.

    Args:
        predictions:   List of prediction dicts (one per document).
        ground_truths: List of ground-truth dicts (one per document).
        fields:        Field names to evaluate.

    Returns:
        Dict mapping field_name → {'precision', 'recall', 'f1', 'exact_match'}.
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "em": 0, "n": 0})

    for pred, gold in zip(predictions, ground_truths):
        for field in fields:
            p_val = pred.get(field)
            g_val = gold.get(field)
            stats[field]["n"] += 1

            if g_val is not None and p_val is not None:
                if exact_match(p_val, g_val):
                    stats[field]["tp"] += 1
                    stats[field]["em"] += 1
                else:
                    stats[field]["fp"] += 1
            elif g_val is not None and p_val is None:
                stats[field]["fn"] += 1
            elif g_val is None and p_val is not None:
                stats[field]["fp"] += 1

    results: Dict[str, Dict[str, float]] = {}
    for field, s in stats.items():
        tp, fp, fn, n = s["tp"], s["fp"], s["fn"], s["n"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        em        = s["em"] / n if n > 0 else 0.0
        results[field] = {
            "precision":    round(precision, 4),
            "recall":       round(recall,    4),
            "f1":           round(f1,        4),
            "exact_match":  round(em,        4),
        }

    return results


def macro_avg(per_field_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged metrics across all fields."""
    keys = ["precision", "recall", "f1", "exact_match"]
    n = len(per_field_metrics)
    if n == 0:
        return {k: 0.0 for k in keys}
    return {k: round(sum(v[k] for v in per_field_metrics.values()) / n, 4) for k in keys}
