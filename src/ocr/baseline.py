import os
import shutil
import sys
import yaml
import pytesseract
from PIL import Image
from src.data.schemas import OCRResult

# Auto-configure Tesseract path on Windows when not in PATH
if sys.platform == "win32" and not shutil.which("tesseract"):
    _win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_path):
        pytesseract.pytesseract.tesseract_cmd = _win_path


def extract_text(image: Image.Image, config_path: str = "configs/ocr_config.yaml") -> OCRResult:
    cfg = _load_config(config_path)
    tesseract_cfg = f"--oem {cfg['oem']} --psm {cfg['psm']}"
    data = pytesseract.image_to_data(
        image,
        lang=cfg["lang"],
        config=tesseract_cfg,
        output_type=pytesseract.Output.DICT,
    )
    words = [w for w in data["text"] if w.strip()]
    text = " ".join(words)
    valid_confs = [c for c in data["conf"] if c != -1]
    avg_conf = (sum(valid_confs) / len(valid_confs) / 100.0) if valid_confs else 0.0
    return OCRResult(text=text, confidence=avg_conf, model="tesseract")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["tesseract"]
