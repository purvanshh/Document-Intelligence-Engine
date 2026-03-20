"""
Training Script
---------------
Fine-tune LayoutLMv3 on annotated document datasets (FUNSD / CORD).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.models.layoutlm_model import LABEL2ID, ID2LABEL
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainer(cfg: Config) -> Trainer:
    """Construct a HuggingFace Trainer with the config settings."""
    set_seed(cfg.seed)

    training_args = TrainingArguments(
        output_dir=str(cfg.checkpoint_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
        logging_dir=str(cfg.log_dir),
        logging_steps=50,
        report_to="none",
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg.base_model,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    return Trainer(
        model=model,
        args=training_args,
        # train_dataset and eval_dataset must be injected by the caller
    )


if __name__ == "__main__":
    # Quick sanity check
    logger.info("Training module loaded successfully.")
