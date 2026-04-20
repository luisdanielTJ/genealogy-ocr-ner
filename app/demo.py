import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import gradio as gr
from PIL import Image
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import run
from src.ocr.baseline import extract_text as tesseract_extract
from src.ocr.evaluate import compute_metrics
from src.data.preprocessing import preprocess

LABEL_COLORS = {
    "NAME": "#FF6B6B",
    "DATE": "#4ECDC4",
    "LOCATION": "#45B7D1",
    "AGE": "#96CEB4",
    "RELATIONSHIP": "#FFEAA7",
}

TESSERACT_CONFIG = PipelineConfig(
    ocr_model="tesseract",
    confidence_threshold=0.6,
    max_sequence_length=128,
)

TROCR_CONFIG = PipelineConfig(
    ocr_model="models/trocr-genealogy",
    confidence_threshold=0.6,
    max_sequence_length=128,
)


def process_image(image: Image.Image):
    if image is None:
        return "No image provided.", "No image provided.", "{}", ""

    preprocessed = preprocess(image)

    tesseract_result = tesseract_extract(preprocessed)
    pipeline_result = run(image, TESSERACT_CONFIG)

    highlighted = _highlight_entities(pipeline_result.raw_text, pipeline_result.entities)

    entities_json = json.dumps(
        [
            {
                "label": e.label,
                "value": e.value,
                "confidence": round(e.confidence, 3),
            }
            for e in pipeline_result.entities
        ],
        indent=2,
    )

    return tesseract_result.text, pipeline_result.raw_text, entities_json, highlighted


def _highlight_entities(text: str, entities) -> str:
    if not entities:
        return f"<p>{text}</p>"
    result = text
    offset = 0
    for entity in sorted(entities, key=lambda e: e.span[0]):
        color = LABEL_COLORS.get(entity.label, "#DDD")
        start = entity.span[0] + offset
        end = entity.span[1] + offset
        original = text[entity.span[0]:entity.span[1]]
        tag = (
            f'<mark style="background:{color};padding:2px 4px;border-radius:3px;'
            f'font-weight:bold" title="{entity.label}">{original}</mark>'
        )
        result = result[:start] + tag + result[end:]
        offset += len(tag) - (entity.span[1] - entity.span[0])
    return f"<p style='font-family:monospace;line-height:2'>{result}</p>"


with gr.Blocks(title="Genealogy Document OCR + NER") as demo:
    gr.Markdown("# Genealogy Document OCR + NER Pipeline")
    gr.Markdown(
        "Upload a historical document or form image to extract structured genealogy entities. "
        "Demonstrates Tesseract baseline vs pipeline output with entity highlighting."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Document Image")

    with gr.Row():
        with gr.Column():
            tesseract_out = gr.Textbox(label="Tesseract Baseline Output", lines=6)
        with gr.Column():
            pipeline_out = gr.Textbox(label="Pipeline OCR Output", lines=6)

    highlighted_out = gr.HTML(label="Extracted Entities (highlighted)")
    entities_json = gr.JSON(label="Structured Entities")

    image_input.change(
        fn=process_image,
        inputs=image_input,
        outputs=[tesseract_out, pipeline_out, entities_json, highlighted_out],
    )

if __name__ == "__main__":
    demo.launch()
