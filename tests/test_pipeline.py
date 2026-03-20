"""
Pipeline Integration Tests
--------------------------
Smoke tests for the major pipeline components.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ── OCR ──────────────────────────────────────────────────────────────────────

class TestOCREngine:
    def test_import(self):
        from src.ocr.ocr_engine import OCREngine
        assert OCREngine is not None

    def test_ocr_token_dataclass(self):
        from src.ocr.ocr_engine import OCRToken
        tok = OCRToken(text="hello", bbox=[0, 0, 100, 50], confidence=0.95)
        assert tok.text == "hello"
        assert tok.confidence == 0.95


# ── Bounding Box ─────────────────────────────────────────────────────────────

class TestBBoxAlignment:
    def test_normalize_bbox(self):
        from src.ocr.bbox_alignment import normalize_bbox
        result = normalize_bbox([100, 200, 300, 400], image_width=1000, image_height=1000)
        assert result == [100, 200, 300, 400]

    def test_iou_identical(self):
        from src.ocr.bbox_alignment import iou
        box = [0, 0, 100, 100]
        assert iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from src.ocr.bbox_alignment import iou
        assert iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0


# ── Postprocessing ────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_date(self):
        from src.postprocessing.validation import validate_output
        result = validate_output({"date": "2025-01-12"})
        assert result["date"] == "2025-01-12"

    def test_invalid_date_cleared(self):
        from src.postprocessing.validation import validate_output
        result = validate_output({"date": "not-a-date"})
        assert result["date"] is None


class TestNormalization:
    def test_parse_amount(self):
        from src.postprocessing.normalization import parse_amount
        assert parse_amount("$1,200.50") == pytest.approx(1200.50)

    def test_parse_date_iso(self):
        from src.postprocessing.normalization import parse_date
        assert parse_date("01/12/2025") in ("2025-01-12", "2025-12-01")  # format-dependent

    def test_correct_ocr_typos(self):
        from src.postprocessing.normalization import correct_ocr_typos
        assert correct_ocr_typos("1OO") == "100"


class TestConstraints:
    def test_no_flag_when_sum_matches(self):
        from src.postprocessing.constraints import apply_constraints
        data = {
            "total_amount": 800.0,
            "line_items": [
                {"item": "A", "price": 400.0, "quantity": 2},
            ],
        }
        result = apply_constraints(data)
        assert result["_constraint_flags"] == []

    def test_flag_when_sum_mismatches(self):
        from src.postprocessing.constraints import apply_constraints
        data = {
            "total_amount": 1000.0,
            "line_items": [
                {"item": "A", "price": 400.0, "quantity": 2},
            ],
        }
        result = apply_constraints(data)
        assert len(result["_constraint_flags"]) > 0


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_score(self):
        from src.evaluation.metrics import compute_field_f1
        preds = [{"invoice_number": "INV-001"}]
        golds = [{"invoice_number": "INV-001"}]
        metrics = compute_field_f1(preds, golds, ["invoice_number"])
        assert metrics["invoice_number"]["f1"] == 1.0

    def test_zero_score(self):
        from src.evaluation.metrics import compute_field_f1
        preds = [{"invoice_number": "WRONG"}]
        golds = [{"invoice_number": "INV-001"}]
        metrics = compute_field_f1(preds, golds, ["invoice_number"])
        assert metrics["invoice_number"]["f1"] == 0.0
