"""End-to-end document processing pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.domain.contracts import DocumentProcessingResult, ValidatedFile
from document_intelligence_engine.ingestion.file_loader import load_pages, persist_validated_file
from document_intelligence_engine.multimodal.layoutlmv3 import LayoutLMv3InferenceService
from document_intelligence_engine.ocr.service import OCRService
from document_intelligence_engine.postprocessing.deterministic import apply_constraints
from document_intelligence_engine.postprocessing.normalizer import normalize_document
from document_intelligence_engine.postprocessing.validator import validate_document
from document_intelligence_engine.preprocessing.image_normalizer import ImageNormalizationService


class DocumentPipeline:
    def __init__(
        self,
        preprocessing_service: ImageNormalizationService | None = None,
        ocr_service: OCRService | None = None,
        inference_service: LayoutLMv3InferenceService | None = None,
    ) -> None:
        self.preprocessing_service = preprocessing_service or ImageNormalizationService()
        self.ocr_service = ocr_service or OCRService()
        self.inference_service = inference_service or LayoutLMv3InferenceService()

    def process(self, document: ValidatedFile) -> DocumentProcessingResult:
        persisted = persist_validated_file(document)
        pages = load_pages(document)

        aggregated_payload: dict[str, object] = {
            "document_id": document.sha256,
            "source_path": str(persisted.path),
            "line_items": [],
        }
        last_engine = "unknown"
        last_model = get_settings().model.layoutlmv3_model_name

        for page in pages:
            normalized_page = self.preprocessing_service.normalize(page)
            ocr_result = self.ocr_service.extract(normalized_page.image_bytes, normalized_page.page_number)
            prediction = self.inference_service.predict(ocr_result)
            aggregated_payload.update(prediction.entities)
            last_engine = ocr_result.engine
            last_model = prediction.model_name

        normalized = normalize_document(aggregated_payload)
        validated = validate_document(normalized)
        constrained = apply_constraints(validated)

        return DocumentProcessingResult(
            document_id=document.sha256,
            status="processed",
            source_name=document.safe_name,
            content_type=document.content_type,
            page_count=len(pages),
            extracted=constrained.normalized,
            constraint_flags=constrained.flags,
            ocr_engine=last_engine,
            model_name=last_model,
            processed_at=datetime.now(timezone.utc),
        )
