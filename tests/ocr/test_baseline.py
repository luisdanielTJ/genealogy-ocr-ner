import pytest
from src.ocr.baseline import extract_text
from src.data.schemas import OCRResult


def test_extract_text_returns_ocr_result(simple_text_image):
    result = extract_text(simple_text_image)
    assert isinstance(result, OCRResult)
    assert result.model == "tesseract"


def test_extract_text_confidence_between_zero_and_one(simple_text_image):
    result = extract_text(simple_text_image)
    assert 0.0 <= result.confidence <= 1.0


def test_extract_text_detects_words(simple_text_image):
    result = extract_text(simple_text_image)
    # Simple clear image — Tesseract should find at least some text
    assert len(result.text.strip()) > 0
