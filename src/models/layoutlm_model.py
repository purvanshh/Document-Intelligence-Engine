"""
LayoutLMv3 Model Wrapper
------------------------
Loads a fine-tuned LayoutLMv3 checkpoint and exposes a clean
predict() interface for token classification (KEY / VALUE / OTHER).
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

LABEL_MAP: Dict[int, str] = {0: "OTHER", 1: "KEY", 2: "VALUE"}
ID2LABEL = LABEL_MAP
LABEL2ID: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}


class LayoutLMModel:
    """
    Thin wrapper around a HuggingFace LayoutLMv3 checkpoint.

    Args:
        model_name_or_path: HF hub ID or local directory.
        device: 'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/layoutlmv3-base",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name_or_path,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)
        self.model.eval()

    def predict(
        self,
        image,  # PIL Image
        words: List[str],
        boxes: List[List[int]],
    ) -> List[Dict]:
        """
        Run inference on a single page.

        Args:
            image: PIL Image of the document page.
            words: Tokenised words from OCR.
            boxes: Corresponding bounding boxes (0-1000 normalised).

        Returns:
            List of dicts: {'word': str, 'box': list, 'label': str, 'score': float}
        """
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        predictions = logits.argmax(dim=-1).cpu().tolist()
        scores = logits.softmax(dim=-1).max(dim=-1).values.cpu().tolist()

        # Map back to word-level predictions (skip special tokens)
        word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else []
        results: List[Dict] = []
        seen: set = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            results.append(
                {
                    "word": words[word_id],
                    "box": boxes[word_id],
                    "label": LABEL_MAP.get(predictions[idx], "OTHER"),
                    "score": round(scores[idx], 4),
                }
            )

        return results
