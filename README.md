# Genealogy Document OCR + NER Pipeline

An end-to-end machine learning pipeline for extracting structured genealogy entities (names, dates, locations, ages, relationships) from historical documents and forms. Built as a portfolio project demonstrating OCR fine-tuning, NER fine-tuning, and production-quality ML engineering.

## What This Demonstrates

- **OCR fine-tuning** — TrOCR (`microsoft/trocr-base-handwritten`) fine-tuned on IAM + FUNSD + synthetic genealogy documents
- **NER fine-tuning** — BERT-NER (`dslim/bert-base-NER`) fine-tuned with BIO tagging for 5 genealogy entity types
- **Baseline comparison** — Tesseract OCR as a reproducible baseline with CER/WER metrics
- **End-to-end pipeline** — image → preprocessing → OCR → NER → structured output
- **Interactive demo** — Gradio app with entity highlighting, deployable to HuggingFace Spaces

## Project Structure

```
ocr/
├── app/
│   └── demo.py               # Gradio interactive demo
├── configs/
│   ├── ocr_config.yaml       # TrOCR training hyperparameters
│   └── ner_config.yaml       # BERT-NER training hyperparameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ocr_baseline.ipynb
│   ├── 03_trocr_finetuning.ipynb
│   ├── 04_ner_finetuning.ipynb
│   └── 05_end_to_end_eval.ipynb
├── src/
│   ├── data/                 # Schemas, loaders, preprocessing, synthetic generation
│   ├── ocr/                  # Tesseract baseline + TrOCR fine-tuning
│   ├── ner/                  # BERT-NER model + fine-tuning
│   └── pipeline/             # End-to-end pipeline config and runner
└── tests/                    # 25 unit tests (pytest)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
git clone <repo>
cd ocr
uv sync --extra dev
```

**Tesseract OCR** must be installed separately:
- Windows: [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) → `C:\Program Files\Tesseract-OCR\`
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

## Usage

### Run the demo

```bash
uv run python app/demo.py
```

Opens a Gradio interface at `http://localhost:7860`. Upload a document image to see:
- Tesseract baseline OCR output
- Pipeline OCR output
- Extracted entities as highlighted HTML and structured JSON

### Run tests

```bash
uv run pytest tests/ -v
```

### Fine-tune models (Kaggle recommended)

The notebooks in `notebooks/` walk through each training stage. Data caps are set to stay within Kaggle's free GPU limits:
- IAM: 2,000 train / 500 val samples
- FUNSD: full dataset (149 train / 50 test documents)
- Synthetic genealogy documents: 1,500 generated

Run on Kaggle with a T4 GPU (~2–3 hours total for both models).

### Pipeline API

```python
from src.pipeline import run, PipelineConfig
from PIL import Image

config = PipelineConfig(ocr_model="tesseract", confidence_threshold=0.5)
result = run(Image.open("document.jpg"), config)

print(result.raw_text)
for entity in result.entities:
    print(f"{entity.label}: {entity.value} ({entity.confidence:.2f})")
```

## Entity Types

| Label | Examples |
|-------|---------|
| NAME | John Smith, Mary O'Brien |
| DATE | 14 March 1872, 1899-07-22 |
| LOCATION | Boston MA, County Cork |
| AGE | 43, age 12 |
| RELATIONSHIP | father, maternal grandmother |

## Metrics

OCR evaluation uses **CER** (Character Error Rate) and **WER** (Word Error Rate) via [jiwer](https://github.com/jitsi/jiwer).

NER evaluation uses **precision**, **recall**, and **F1** per entity type and macro-averaged via [seqeval](https://github.com/chakki-works/seqeval).

## Tech Stack

- **Models**: HuggingFace Transformers, TrOCR, BERT
- **OCR baseline**: Tesseract via pytesseract
- **Data**: IAM Handwriting DB, FUNSD, Faker-generated synthetic documents
- **Training**: Seq2SeqTrainer (OCR), Trainer (NER)
- **Demo**: Gradio
- **Package management**: uv
- **Testing**: pytest (25 tests)
