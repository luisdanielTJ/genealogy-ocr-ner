from PIL import Image
from src.data.preprocessing import preprocess
from src.data.schemas import PipelineResult, Entity
from src.pipeline.config import PipelineConfig


def run(image: Image.Image, config: PipelineConfig) -> PipelineResult:
    preprocessed = preprocess(image)
    ocr_result = _run_ocr(preprocessed, config)
    entities = _run_ner(ocr_result.text, config)
    entities = [e for e in entities if e.confidence >= config.confidence_threshold]
    return PipelineResult(
        raw_text=ocr_result.text,
        entities=entities,
        ocr_confidence=ocr_result.confidence,
        model_used=ocr_result.model,
    )


def _run_ocr(image: Image.Image, config: PipelineConfig):
    if config.ocr_model == "tesseract":
        from src.ocr.baseline import extract_text
        return extract_text(image)
    else:
        from src.ocr.trocr_finetune import load_trocr
        extract_text = load_trocr(config.ocr_model)
        return extract_text(image)


def _run_ner(text: str, config: PipelineConfig) -> list[Entity]:
    try:
        from src.ner.model import load_finetuned, extract_entities
        model, tokenizer, id2label = load_finetuned(
            "models/ner-genealogy", config_path="configs/ner_config.yaml"
        )
        return extract_entities(text, model, tokenizer, id2label, config.max_sequence_length)
    except (OSError, Exception):
        # Fine-tuned model not yet available — return empty until trained on Kaggle
        return []
