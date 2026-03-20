"""LayoutLMv3 training hook definitions."""

from __future__ import annotations

from pydantic import BaseModel

from document_intelligence_engine.core.config import get_settings


class TrainingHookSpec(BaseModel):
    model_name: str
    batch_size: int
    max_sequence_length: int
    revision: str


def build_training_hook_spec() -> TrainingHookSpec:
    settings = get_settings()
    return TrainingHookSpec(
        model_name=settings.model.layoutlmv3_model_name,
        batch_size=settings.model.batch_size,
        max_sequence_length=settings.model.max_sequence_length,
        revision=settings.model.revision,
    )
