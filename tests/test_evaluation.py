from __future__ import annotations

import json
from pathlib import Path

from evaluation.ablation import run_ablation_study
from evaluation.benchmark import run_benchmark
from evaluation.error_analysis import analyze_errors
from evaluation.report import generate_report


class DummyModel:
    name = "dummy-layoutlmv3"

    def predict(self, ocr_tokens):
        _ = ocr_tokens
        return [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.98},
            {"text": "Number", "label": "I-KEY", "confidence": 0.97},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.96},
            {"text": "Date", "label": "B-KEY", "confidence": 0.95},
            {"text": "2025-01-12", "label": "B-VALUE", "confidence": 0.94},
            {"text": "Total", "label": "B-KEY", "confidence": 0.96},
            {"text": "Amount", "label": "I-KEY", "confidence": 0.95},
            {"text": "1200.50", "label": "B-VALUE", "confidence": 0.93},
        ]

    def predict_text_only(self, ocr_tokens):
        _ = ocr_tokens
        return [
            {"text": "Invoice", "label": "B-KEY", "confidence": 0.90},
            {"text": "Number", "label": "I-KEY", "confidence": 0.90},
            {"text": "INV-1023", "label": "B-VALUE", "confidence": 0.89},
        ]

    def predict_without_postprocessing(self, ocr_tokens):
        return self.predict(ocr_tokens)


class DummyPipeline:
    def postprocess_predictions(self, predictions):
        from postprocessing.pipeline import postprocess_predictions

        return postprocess_predictions(predictions)


def test_run_benchmark_and_report(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "document_path": str(tmp_path / "sample.pdf"),
                    "token_labels": [
                        "B-KEY",
                        "I-KEY",
                        "B-VALUE",
                        "B-KEY",
                        "B-VALUE",
                        "B-KEY",
                        "I-KEY",
                        "B-VALUE",
                    ],
                    "ground_truth": {
                        "invoice_number": {"value": "INV-1023"},
                        "date": {"value": "2025-01-12"},
                        "total_amount": {"value": 1200.5},
                    },
                    "entities": [
                        {"field": "invoice_number", "value": "INV-1023"},
                        {"field": "date", "value": "2025-01-12"},
                        {"field": "total_amount", "value": 1200.5},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "evaluation.benchmark.process_document",
        lambda path: [
            {"text": "Invoice", "bbox": [0, 0, 1, 1], "confidence": 0.99},
            {"text": "Number", "bbox": [0, 0, 1, 1], "confidence": 0.98},
            {"text": "INV-1023", "bbox": [0, 0, 1, 1], "confidence": 0.97},
            {"text": "Date", "bbox": [0, 0, 1, 1], "confidence": 0.96},
            {"text": "2025-01-12", "bbox": [0, 0, 1, 1], "confidence": 0.95},
            {"text": "Total", "bbox": [0, 0, 1, 1], "confidence": 0.94},
            {"text": "Amount", "bbox": [0, 0, 1, 1], "confidence": 0.93},
            {"text": "1200.50", "bbox": [0, 0, 1, 1], "confidence": 0.92},
        ],
    )
    monkeypatch.setenv("DIE_EVALUATION__TRACKING__ENABLED", "false")

    result = run_benchmark(str(dataset_path), DummyModel(), DummyPipeline())
    assert result["sample_count"] == 1
    assert "full_system" in result["baselines"]

    ablation = run_ablation_study(str(dataset_path), DummyModel(), DummyPipeline())
    assert ablation["comparison_table"]

    errors = analyze_errors(result)
    assert "ocr_errors" in errors

    report = generate_report(result, ablation_results=ablation, output_dir=str(tmp_path / "reports"))
    assert Path(report["json_path"]).exists()
    assert Path(report["markdown_path"]).exists()
