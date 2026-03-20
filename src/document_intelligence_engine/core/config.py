"""Centralized application configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
ENV_PREFIX = "DIE_"


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    annotations_dir: Path
    experiment_dir: Path
    artifact_dir: Path
    upload_dir: Path


class APIConfig(BaseModel):
    host: str
    port: int
    reload: bool
    workers: int
    cors_origins: list[str]
    max_upload_size_mb: int


class OCRConfig(BaseModel):
    backend: str
    language: str
    min_confidence: float
    tesseract_cmd: str | None = None


class ModelConfig(BaseModel):
    layoutlmv3_model_name: str
    revision: str
    device: str
    max_sequence_length: int
    batch_size: int


class PreprocessingConfig(BaseModel):
    target_dpi: int
    max_image_side: int
    grayscale: bool


class LoggingConfig(BaseModel):
    level: str
    json: bool
    service_name: str

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log level")
        return normalized


class SecurityConfig(BaseModel):
    allowed_extensions: list[str]
    allowed_content_types: list[str]
    max_pdf_pages: int
    max_image_pixels: int
    filename_pattern: str = Field(
        default=r"[^A-Za-z0-9._-]",
        description="Pattern replaced during filename sanitization.",
    )


class AppSettings(BaseModel):
    project_name: str
    environment: str
    debug: bool
    paths: PathsConfig
    api: APIConfig
    ocr: OCRConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    logging: LoggingConfig
    security: SecurityConfig


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_pointer:
        return yaml.safe_load(file_pointer) or {}


def _coerce_env_value(raw_value: str, current_value: Any) -> Any:
    if isinstance(current_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, list):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(current_value, Path):
        return Path(raw_value)
    return raw_value


def _apply_env_overrides(config: dict[str, Any], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        path_parts = env_key[len(prefix) :].lower().split("__")
        target = config
        for part in path_parts[:-1]:
            target = target.setdefault(part, {})

        final_key = path_parts[-1]
        current_value = target.get(final_key, raw_value)
        target[final_key] = _coerce_env_value(raw_value, current_value)

    return config


def _resolve_paths(settings: AppSettings) -> AppSettings:
    for field_name, value in settings.paths.model_dump().items():
        path_value = Path(value)
        resolved = path_value if path_value.is_absolute() else ROOT_DIR / path_value
        setattr(settings.paths, field_name, resolved)
        resolved.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> AppSettings:
    path = config_path or DEFAULT_CONFIG_PATH
    config_data = _apply_env_overrides(_load_yaml_config(path))
    settings = AppSettings.model_validate(config_data)
    return _resolve_paths(settings)
