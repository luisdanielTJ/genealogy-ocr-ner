from src.ocr.evaluate import compute_metrics, compare_models
from src.data.schemas import OCRResult


def test_perfect_prediction_gives_zero_cer():
    metrics = compute_metrics(["hello world"], ["hello world"])
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0


def test_wrong_prediction_gives_nonzero_cer():
    metrics = compute_metrics(["helo world"], ["hello world"])
    assert metrics["cer"] > 0.0


def test_compare_models_returns_both_keys():
    tesseract = [OCRResult(text="hello world", confidence=0.8, model="tesseract")]
    trocr = [OCRResult(text="hello world", confidence=0.95, model="trocr-genealogy")]
    refs = ["hello world"]
    result = compare_models(tesseract, trocr, refs)
    assert "tesseract" in result
    assert "trocr" in result
    assert "cer" in result["tesseract"]
    assert "wer" in result["trocr"]
