"""
Config Module
-------------
Central configuration loaded from environment variables + .env file.
Uses Pydantic Settings for typed, validated configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model
    base_model: str = "microsoft/layoutlmv3-base"
    checkpoint_dir: Path = Path("experiments/checkpoints")

    # Training
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 5e-5
    seed: int = 42

    # OCR
    ocr_backend: str = "paddle"   # "paddle" | "tesseract"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_dir: Path = Path("experiments/logs")

    # Data paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    annotations_dir: Path = Path("data/annotations")

    # MLOps (optional)
    mlflow_tracking_uri: Optional[str] = None
    wandb_project: Optional[str] = None


# Singleton instance
cfg = Config()
# For backwards compatibility you can also import as `Config`
