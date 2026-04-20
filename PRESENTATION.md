# Ancestry.com Interview — Project Presentation Guide

## Elevator Pitch (30 seconds)

> "I built an end-to-end ML pipeline that extracts structured genealogy entities — names, dates, locations, ages, relationships — from historical documents. It fine-tunes two models: TrOCR for handwriting recognition, which achieves a 7x improvement in character accuracy over Tesseract, and BERT-NER for entity extraction, which reaches F1=1.0 on genealogy fields. The whole thing runs in a Gradio demo and is backed by 25 unit tests."

---

## The Three Skills This Demonstrates

| Job Requirement | What the Project Shows |
|---|---|
| Fine-tuning models | TrOCR fine-tuned on IAM + FUNSD + synthetic data |
| OCR document extraction | Tesseract baseline vs fine-tuned TrOCR with CER/WER metrics |
| NLP / NER | BERT-NER fine-tuned for 5 genealogy entity types with per-label F1 |

---

## Project Architecture

```
Image
  ↓
Preprocessing        ← grayscale, Otsu binarization, deskew, denoise (OpenCV)
  ↓
OCR                  ← Tesseract (baseline) or fine-tuned TrOCR
  ↓
NER                  ← fine-tuned BERT extracts entities from OCR text
  ↓
Structured Output    ← {NAME, DATE, LOCATION, AGE, RELATIONSHIP}
```

**Key design principle:** Each layer is independently testable and swappable. OCR model can be changed via config without touching NER code.

---

## The Two Models

### OCR — TrOCR
- **Base model:** `microsoft/trocr-base-handwritten`
- **Architecture:** Vision Transformer (ViT) image encoder + RoBERTa text decoder
- **Fine-tuned on:** IAM Handwriting DB (2000 samples) + FUNSD forms (149 docs) + 1500 synthetic genealogy documents
- **Training time:** ~38 minutes on Kaggle T4 GPU
- **Results:**

| Model | CER | WER |
|---|---|---|
| Tesseract baseline | 0.3779 | 0.8022 |
| Fine-tuned TrOCR | 0.0509 | 0.1284 |
| **Improvement** | **7.4x** | **6.2x** |

- **Key limitation:** TrOCR processes single lines. Full-page documents need line segmentation first (e.g., CRAFT text detector). For printed text, Tesseract remains more reliable — model selection should match input distribution.

### NER — BERT
- **Base model:** `dslim/bert-base-NER`
- **Architecture:** BERT-base (12 layers, 110M parameters), pretrained on CoNLL-2003 NER
- **Fine-tuned on:** 1500 synthetic genealogy documents (BIO tagging scheme, 11 labels)
- **Training time:** ~38 minutes on Kaggle T4 GPU
- **Results:** Precision: 1.000 / Recall: 1.000 / F1: 1.000 on synthetic validation set
- **Entity types:** NAME, DATE, LOCATION, AGE, RELATIONSHIP

**Important distinction:** BERT is not an OCR model — it never sees images. It only processes text output from TrOCR/Tesseract. Format (handwritten vs printed) is irrelevant to BERT.

---

## Datasets

| Dataset | Type | Use | Size used |
|---|---|---|---|
| IAM Handwriting DB | Real handwritten lines | TrOCR training | 2000 train / 500 val |
| FUNSD | Scanned forms | TrOCR training | 149 train / 50 test |
| Synthetic (Faker + Pillow) | Generated genealogy forms | NER training | 1500 docs |

**Why synthetic data for NER:** Generating documents programmatically gives exact character-level span annotations for free — no human labeling needed. Ground truth is built in.

---

## Evaluation Metrics

| Model | Metrics | Why |
|---|---|---|
| OCR (TrOCR / Tesseract) | CER (Character Error Rate), WER (Word Error Rate) | Measures how accurately text is transcribed |
| NER (BERT) | Precision, Recall, F1 per entity type | Measures how well entities are found and classified |

**CER vs WER:**
- CER = character-level errors / total characters (more granular)
- WER = word-level errors / total words (more intuitive)
- A single wrong character ruins the whole word in WER but only counts once in CER

---

## Tech Stack

| Component | Technology |
|---|---|
| Models | HuggingFace Transformers |
| OCR baseline | Tesseract via pytesseract |
| Training | Seq2SeqTrainer (TrOCR), Trainer (BERT) |
| Data | HuggingFace Datasets, Faker, Pillow |
| Preprocessing | OpenCV |
| Evaluation | jiwer (CER/WER), seqeval (NER F1) |
| Demo | Gradio |
| Package management | uv |
| Testing | pytest (25 tests) |
| GPU training | Kaggle T4 |

---

## Production Thinking

### Scaling to Ancestry's volume

For thousands of concurrent document uploads the architecture would be:

```
Upload → API Gateway → Message Queue → Worker Fleet → Results DB → Notify user
```

**Key insight:** Processing is async — users upload and get notified when done. This decouples upload spikes from processing capacity.

**GPU/CPU split:**
- TrOCR requires GPU (expensive) — scale carefully
- BERT NER runs on CPU (cheap) — scale liberally
- Only pay for GPUs where you actually need them

**Batch inference:** Processing 32 documents at once on one GPU costs nearly the same as processing 1. Batching is the biggest cost lever.

### Why fine-tuning is the right approach (not training from scratch)

- Training from scratch requires billions of examples, months of compute, hundreds of GPUs
- Fine-tuning gives 90%+ of that performance at a fraction of the cost
- At Ancestry's scale you'd use larger base models (`trocr-large`, `roberta-large`) but the methodology is identical

### The data labeling advantage

**The most expensive part of ML at scale is not GPU — it's data labeling.**

A professional annotation service charges $0.05–$2.00 per label. At 1 million labels that's $50K–$2M just for labeling.

**Ancestry's structural advantage:**
- 20+ years of user transcriptions = millions of free labeled examples
- User corrections = hard negatives with correct answers attached
- Structured record databases can be matched back to document images as ground truth
- Every user interaction is a potential training signal — a data flywheel

> "Ancestry doesn't have a data labeling problem — they have a data engineering problem. The challenge is building the pipeline to clean, align, and feed that existing data into model training."

### More data vs more epochs

| Situation | What to do |
|---|---|
| Loss still decreasing at end of training | More epochs (model hasn't converged) |
| Loss flat but accuracy still poor | More data (model hit a ceiling) |
| Great training metrics, poor real-world results | More diverse data (overfitting) |

More data improves generalization — the right long-term investment. More epochs squeezes more out of existing data — the cheap short-term fix.

### Handling multiple document formats

**You don't need one model per format.** Two options:

1. **Mixed training** — train on all formats combined. Simpler, one model to serve.
2. **Specialist models + router** — classify document type first, route to specialist. Better when formats are very different (1800s cursive vs modern printed forms).

For Ancestry specifically, the likely architecture is a **document classifier** upfront:

```
Image → Document Classifier → OCR specialist → NER specialist → Entities
```

Each document type (US Census 1880, Irish parish record, WWII draft card) gets a model trained on its specific format, handwriting era, and field structure.

---

## What to Show in the Demo

1. **Upload a synthetic test image** (`test_document.png`) — shows clean entity extraction on structured forms
2. **Upload a handwritten IAM sample** (`test_handwritten_0.png` etc.) — shows TrOCR vs Tesseract difference
3. **Point to the JSON panel** — shows structured output, confidence scores, entity filtering at 0.6 threshold
4. **Open notebook 02** — shows Tesseract baseline CER/WER numbers
5. **Open notebook 03** — shows the 7x improvement bar chart
6. **Open notebook 04** — shows per-entity F1 table and chart

---

## Key Talking Points

**On the baseline comparison:**
> "Tesseract misreads 38% of characters on handwriting — essentially unusable for genealogy records. After fine-tuning TrOCR, CER drops to 5%. That's the difference between garbage output and something a downstream NER model can actually work with."

**On the NER model:**
> "We started from dslim/bert-base-NER which already understands what entities look like from CoNLL-2003 training. We only needed to teach it genealogy-specific types — domain adaptation rather than training from scratch."

**On synthetic data:**
> "Generating synthetic documents programmatically gives exact span annotations for free. The ground truth is built in — no human labeling needed for the NER training phase."

**On model selection:**
> "TrOCR excels on handwriting but struggles on full pages since it's trained on single lines. For printed forms, Tesseract is actually more reliable. In production you'd route documents based on type — that's a natural extension of this architecture."

**On production readiness:**
> "The architecture is modular — OCR and NER are independent layers connected by a clean interface. Swapping the OCR model from Tesseract to TrOCR is a one-line config change. That's what makes it extensible to Ancestry's document variety."

**On Ancestry's data advantage:**
> "The methodology scales directly to Ancestry's corpus. They have the original documents and the ground truth from user transcriptions — the same structure we used with synthetic data. The training pipeline stays identical, only the data loader changes."

---

## Anticipated Interview Questions

**Q: Why not use GPT-4 Vision for OCR instead of TrOCR?**
> GPT-4V works but it's expensive at scale ($0.01+ per image), non-deterministic, and can't be fine-tuned on your specific document corpus. TrOCR is cheaper, faster, deterministic, and improves with your data.

**Q: How would you improve the model further?**
> More training data is always the primary lever. For TrOCR: more epochs (loss was still decreasing), larger base model (trocr-large), line segmentation for full-page documents. For NER: more diverse synthetic data, real Ancestry documents, handling multi-language records.

**Q: How do you handle documents in other languages?**
> The current pipeline is English-only. For multi-language: multilingual BERT (bert-base-multilingual-cased) for NER, language detection upfront to route to the right model, and language-specific training data.

**Q: What's the biggest risk in production?**
> Distribution shift — the model performs well on the training distribution but degrades on document types it hasn't seen. Mitigation: continuous evaluation on production data, confidence thresholds to flag low-confidence extractions for human review, and regular retraining as new document types are added.

**Q: How do you evaluate NER in production without ground truth?**
> Confidence scores flag uncertain predictions. You sample low-confidence outputs for human review, which also generates more labeled data. Over time this becomes a human-in-the-loop retraining cycle.
