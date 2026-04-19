from src.data.schemas import Entity, Document, OCRResult, PipelineResult
from PIL import Image


def test_entity_creation():
    entity = Entity(label="NAME", value="John Smith", confidence=0.95, span=(0, 10))
    assert entity.label == "NAME"
    assert entity.value == "John Smith"
    assert entity.confidence == 0.95
    assert entity.span == (0, 10)


def test_document_creation():
    img = Image.new("RGB", (100, 100), color="white")
    doc = Document(image=img, text="John Smith", entities=[], source="synthetic")
    assert doc.source == "synthetic"
    assert doc.text == "John Smith"
    assert doc.entities == []


def test_ocr_result_creation():
    result = OCRResult(text="hello", confidence=0.9, model="tesseract")
    assert result.model == "tesseract"


def test_pipeline_result_creation():
    result = PipelineResult(raw_text="hello", entities=[], ocr_confidence=0.9, model_used="tesseract")
    assert result.model_used == "tesseract"
