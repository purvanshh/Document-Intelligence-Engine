# 🧠 Document Intelligence Engine

> **Layout-Aware Multimodal Document Parsing** — Converting PDFs & images into validated, structured JSON with deterministic post-processing.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-LayoutLMv3-FFD21E?style=flat)](https://huggingface.co/microsoft/layoutlmv3-base)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem Statement

Existing document extraction tools fail in one of two ways:

| Approach | Failure Mode |
|---|---|
| Rule-based / template | Breaks on any layout variation |
| LLM-based extraction | Non-deterministic, unreliable for production |

This engine combines **LayoutLMv3** (vision + text + layout) with a **deterministic post-processing layer** to deliver production-grade, validated structured extractions every time.

---

## 🏗️ System Architecture

```
PDF / Image
    │
    ▼
┌─────────────────────────────┐
│  Ingestion & Preprocessing  │  Skew correction, normalisation
└──────────────┬──────────────┘
               │
    ▼
┌─────────────────────────────┐
│       OCR Engine            │  PaddleOCR (primary) / Tesseract
│   tokens + bounding boxes   │
└──────────────┬──────────────┘
               │
    ▼
┌─────────────────────────────┐
│    LayoutLMv3 (HuggingFace) │  Token classification
│   KEY · VALUE · OTHER       │  KEY / VALUE / OTHER labels
└──────────────┬──────────────┘
               │
    ▼
┌─────────────────────────────┐
│  Deterministic Post-Proc.   │  Regex validation, normalisation,
│  (Non-negotiable layer)     │  cross-field constraints, confidence
└──────────────┬──────────────┘
               │
    ▼
┌─────────────────────────────┐
│   Structured JSON Output    │  + constraint flags + confidence
└─────────────────────────────┘
               │
    ▼
  FastAPI  /parse-document
```

---

## ✨ Key Features

- **Multimodal**: Jointly reasons over pixel layout, OCR tokens, and bounding boxes
- **Deterministic**: Post-processing layer enforces field-level validation and cross-field constraints — same input, guaranteed same output
- **Production API**: FastAPI with async support, file upload, and health checks
- **Dockerised**: Single `docker compose up` to run the full stack
- **Ablation-ready**: Three built-in ablation experiments to quantify each component's contribution
- **Extensible**: Swap OCR backends, model checkpoints, or post-processing rules without touching the pipeline

---

## 📂 Repository Structure

```
document-intelligence-engine/
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PDF → page image arrays (PyMuPDF)
│   │   └── image_preprocessing.py # Skew correction, normalisation
│   │
│   ├── ocr/
│   │   ├── ocr_engine.py          # PaddleOCR + Tesseract wrapper
│   │   └── bbox_alignment.py      # BBox normalisation, IoU, reading order
│   │
│   ├── models/
│   │   ├── layoutlm_model.py      # LayoutLMv3 inference wrapper
│   │   ├── training.py            # HuggingFace Trainer setup
│   │   └── inference.py           # End-to-end InferencePipeline
│   │
│   ├── postprocessing/
│   │   ├── validation.py          # Regex field validators
│   │   ├── normalization.py       # Date/currency normalisation, OCR typo fix
│   │   └── constraints.py         # Cross-field consistency checks
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Precision / Recall / F1 / Exact Match
│   │   └── ablation.py            # 3 ablation experiments
│   │
│   ├── api/
│   │   ├── main.py                # FastAPI app + CORS
│   │   └── routes.py              # POST /parse-document
│   │
│   └── utils/
│       ├── config.py              # Pydantic Settings (env-driven)
│       └── logger.py              # Structured logging
│
├── data/
│   ├── raw/                       # Original PDFs / images (gitignored)
│   ├── processed/                 # Preprocessed data (gitignored)
│   └── annotations/               # Ground-truth labels (gitignored)
│
├── notebooks/
│   ├── exploration.ipynb          # EDA and data exploration
│   └── experiments.ipynb          # Ablation + results visualisation
│
├── experiments/
│   ├── logs/                      # Training logs
│   └── checkpoints/               # Model checkpoints (gitignored)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   └── test_pipeline.py           # Pytest suite (OCR, postproc, metrics)
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/purvanshh/document-intelligence-engine.git
cd document-intelligence-engine

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set BASE_MODEL, OCR_BACKEND, etc.
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Parse a Document

```bash
curl -X POST http://localhost:8000/parse-document \
     -F "file=@invoice.pdf"
```

---

## 🐳 Docker

```bash
# Build and start
cd docker
docker compose up --build

# With MLflow tracking
docker compose --profile mlops up
```

---

## 📊 Output Schema

```json
{
  "invoice_number": "INV-1023",
  "date": "2025-01-12",
  "vendor": "ABC Pvt Ltd",
  "total_amount": 1200.50,
  "line_items": [
    { "item": "Product A", "quantity": 2, "price": 400.0 }
  ],
  "confidence": {
    "invoice_number": 0.92,
    "total_amount": 0.88
  },
  "_constraint_flags": []
}
```

`_constraint_flags` is populated when cross-field checks fail (e.g., `line_items_sum_mismatch`).

---

## 🧪 Evaluation & Ablation

### Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Target Metrics

| Metric | Target |
|---|---|
| Key-Value Extraction F1 | ≥ 0.80 |
| Exact Match Accuracy | ≥ 0.70 |
| OCR Error Recovery | +15–25% vs raw OCR |
| API Latency | < 2s per document |

### Ablation Studies

Three experiments are implemented in `src/evaluation/ablation.py`:

| Experiment | Purpose |
|---|---|
| **No layout embeddings** | Quantifies importance of spatial bounding boxes |
| **OCR-only baseline** | Measures gain from the full multimodal pipeline |
| **No post-processing** | Shows deterministic layer's impact on accuracy |

---

## 🧠 Model Fine-Tuning

```bash
# Configure in .env then run:
python -m src.models.training
```

Datasets: [FUNSD](https://guillaumejaume.github.io/FUNSD/) · [CORD](https://github.com/clovaai/cord)

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Backbone model | LayoutLMv3 (HuggingFace Transformers) |
| Deep learning | PyTorch 2.x |
| OCR | PaddleOCR / Tesseract |
| Image processing | OpenCV, Pillow, PyMuPDF |
| API | FastAPI + Uvicorn |
| Configuration | Pydantic Settings |
| Containerisation | Docker + Docker Compose |
| MLOps (optional) | MLflow |
| Testing | Pytest |

---

## 📋 Milestones

- [ ] **Week 1** — Dataset setup, OCR pipeline, LayoutLMv3 fine-tuning
- [ ] **Week 2** — Evaluation metrics, ablation studies, post-processing layer
- [ ] **Week 3** — API, Docker deployment, demo UI, README polish

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -m 'feat: add my feature'`)
4. Push and open a PR

---

## 📄 License

MIT © [Purvansh](https://github.com/purvanshh)
