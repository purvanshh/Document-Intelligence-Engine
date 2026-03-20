# Layout-Aware Document Intelligence Engine

## 1. HLD

### Architecture Diagram

```text
Clients / Batch Jobs
        |
        v
+------------------------------+
| API Layer                    |
| FastAPI app, request schema, |
| upload validation, errors    |
+---------------+--------------+
                |
                v
+---------------+--------------+
| Ingestion Layer              |
| file validation, safe store, |
| PDF/image page loading       |
+---------------+--------------+
                |
                v
+---------------+--------------+
| Preprocessing Layer          |
| image normalization, resize, |
| orientation-safe transforms  |
+---------------+--------------+
                |
                v
+---------------+--------------+
| OCR Layer                    |
| text, bounding boxes,        |
| confidence extraction        |
+---------------+--------------+
                |
                v
+---------------+--------------+
| Multimodal Model Layer       |
| LayoutLMv3 inference,        |
| training hooks              |
+---------------+--------------+
                |
                v
+---------------+--------------+
| Post-processing Layer        |
| normalization, validation,   |
| deterministic constraints    |
+---------------+--------------+
                |
                v
+---------------+--------------+
| Evaluation Layer             |
| field metrics, exact match,  |
| ablation definitions         |
+---------------+--------------+
                |
                v
Deterministic Structured JSON

Infrastructure Layer: configs, env overrides, Docker, logging, tests
```

### Data Flow

```text
UploadFile
-> validate_upload()
-> persist_validated_file()
-> load_pages()
-> ImageNormalizationService.normalize()
-> OCRService.extract()
-> LayoutLMv3InferenceService.predict()
-> normalize_document()
-> validate_document()
-> apply_constraints()
-> DocumentParseResponse
```

### Module Interfaces

```text
ingestion.validators.validate_upload(upload_file: UploadFile) -> ValidatedFile
ingestion.file_loader.load_pages(document: ValidatedFile) -> list[IngestedPage]
preprocessing.image_normalizer.ImageNormalizationService.normalize(page: IngestedPage) -> IngestedPage
ocr.service.OCRService.extract(image_bytes: bytes, page_number: int) -> OCRResult
multimodal.layoutlmv3.LayoutLMv3InferenceService.predict(ocr_result: OCRResult) -> ModelPrediction
postprocessing.normalizer.normalize_document(payload: dict[str, object]) -> dict[str, object]
postprocessing.validator.validate_document(payload: dict[str, object]) -> dict[str, object]
postprocessing.deterministic.apply_constraints(payload: dict[str, object]) -> ConstraintResult
services.pipeline.DocumentPipeline.process(document: ValidatedFile) -> DocumentProcessingResult
```

## 2. LLD

### Module Breakdown

```text
src/document_intelligence_engine/api/app.py
  FastAPI factory, router registration, exception mapping

src/document_intelligence_engine/api/routes/health.py
  Health endpoint

src/document_intelligence_engine/api/routes/documents.py
  Parse endpoint, upload-to-pipeline orchestration

src/document_intelligence_engine/api/schemas/
  Strict request/response models

src/document_intelligence_engine/core/config.py
  YAML config loading, env override merge, typed settings

src/document_intelligence_engine/core/logging.py
  Root logger initialization, JSON/plain formatter selection

src/document_intelligence_engine/core/errors.py
  Domain-specific exception hierarchy

src/document_intelligence_engine/domain/contracts.py
  Typed contracts for OCR, model, file, page, output payloads

src/document_intelligence_engine/ingestion/validators.py
  File type checks, signature validation, size limits, sanitization, malformed file rejection

src/document_intelligence_engine/ingestion/file_loader.py
  Safe persistence, PDF page rasterization, image loading

src/document_intelligence_engine/preprocessing/image_normalizer.py
  Deterministic page normalization

src/document_intelligence_engine/ocr/base.py
  OCR backend protocol

src/document_intelligence_engine/ocr/service.py
  Tesseract backend wrapper, OCR service boundary

src/document_intelligence_engine/multimodal/layoutlmv3.py
  LayoutLMv3 inference boundary

src/document_intelligence_engine/multimodal/training.py
  Training hook specification

src/document_intelligence_engine/postprocessing/normalizer.py
  Date/amount/string normalization

src/document_intelligence_engine/postprocessing/validator.py
  Field-level validators

src/document_intelligence_engine/postprocessing/deterministic.py
  Cross-field deterministic constraints

src/document_intelligence_engine/evaluation/metrics.py
  Exact match and field accuracy

src/document_intelligence_engine/evaluation/ablations.py
  Canonical ablation definitions

src/document_intelligence_engine/services/pipeline.py
  End-to-end pipeline orchestration
```

### Contracts, Errors, Logging

```text
Contracts
  ValidatedFile: sanitized upload metadata + raw bytes
  IngestedPage: page image bytes + dimensions + page number
  OCRResult: OCRToken list + engine metadata
  ModelPrediction: model labels + confidences + extracted entities
  ConstraintResult: normalized output + flags
  DocumentProcessingResult: final response contract

Error Strategy
  InvalidInputError -> HTTP 400
  OCRProcessingError -> HTTP 502
  ModelInferenceError -> HTTP 502
  DocumentEngineError -> HTTP 500

Logging Strategy
  JSON logs to stdout
  level controlled by config
  module logger access through get_logger()
```

## 3. Repo Structure

```text
.
├── configs/                        # Centralized YAML configuration
├── data/                           # Raw, processed, and annotation datasets
│   ├── raw/                        # Source PDFs/images
│   ├── processed/                  # Derived intermediate artifacts
│   └── annotations/                # Ground-truth labels
├── docker/                         # Container build and compose assets
├── experiments/                    # Experiment outputs and run artifacts
│   ├── runs/                       # Run metadata and tracking outputs
│   └── artifacts/                  # Checkpoints and exported artifacts
├── src/                            # Application source root
│   └── document_intelligence_engine/
│       ├── api/                    # FastAPI app, routes, schemas
│       ├── core/                   # Config, logging, errors
│       ├── domain/                 # Typed data contracts
│       ├── ingestion/              # File validation and page loading
│       ├── preprocessing/          # Image normalization
│       ├── ocr/                    # OCR interfaces and backends
│       ├── multimodal/             # LayoutLMv3 inference and training hooks
│       ├── postprocessing/         # Normalization, validation, constraints
│       ├── evaluation/             # Metrics and ablations
│       └── services/               # End-to-end pipeline orchestration
└── tests/                          # Unit and integration test suites
```

## 4. Config System

```text
Primary config: configs/config.yaml
Loader: src/document_intelligence_engine/core/config.py
Env override prefix: DIE_
Nested override format: DIE_<SECTION>__<FIELD>=value
Example: DIE_API__PORT=8080
```

## 5. Logging System

```text
Module: src/document_intelligence_engine/core/logging.py
Default sink: stdout
Formats: JSON or plain text
Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 6. Security Setup

```text
Allowed file types: PDF, PNG, JPEG, TIFF
Controls:
  extension validation
  MIME validation
  magic-number validation
  max upload size enforcement
  max PDF page enforcement
  max image pixel enforcement
  filename sanitization
  malformed PDF/image rejection
  non-root Docker runtime
```

## 7. requirements.txt

```text
Python version: 3.11.11
Virtual environment:
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
```

See `requirements.txt` for the fully pinned dependency set.

## 8. .env Template

See `.env.example`.

## 9. Starter Code Files

```text
FastAPI app: src/document_intelligence_engine/api/app.py
Config loader: src/document_intelligence_engine/core/config.py
Logger setup: src/document_intelligence_engine/core/logging.py
Entrypoint: src/document_intelligence_engine/entrypoint.py
```
