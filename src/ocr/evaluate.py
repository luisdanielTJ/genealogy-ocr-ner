from jiwer import cer, wer
from src.data.schemas import OCRResult


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    return {
        "cer": cer(references, predictions),
        "wer": wer(references, predictions),
    }


def compare_models(
    tesseract_results: list[OCRResult],
    trocr_results: list[OCRResult],
    references: list[str],
) -> dict:
    return {
        "tesseract": compute_metrics([r.text for r in tesseract_results], references),
        "trocr": compute_metrics([r.text for r in trocr_results], references),
    }
