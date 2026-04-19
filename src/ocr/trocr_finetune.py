import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from src.data.schemas import Document, OCRResult


class _OCRDataset(Dataset):
    def __init__(self, documents: list[Document], processor: TrOCRProcessor, max_length: int):
        self.documents = documents
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict:
        doc = self.documents[idx]
        pixel_values = self.processor(doc.image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            doc.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def finetune(
    train_docs: list[Document],
    val_docs: list[Document],
    config_path: str = "configs/ocr_config.yaml",
) -> str:
    """Fine-tunes TrOCR on train_docs, evaluates on val_docs. Returns output_dir."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["trocr"]

    processor = TrOCRProcessor.from_pretrained(cfg["base_model"])
    model = VisionEncoderDecoderModel.from_pretrained(cfg["base_model"])
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    t = cfg["training"]
    train_ds = _OCRDataset(train_docs, processor, t["max_target_length"])
    val_ds = _OCRDataset(val_docs, processor, t["max_target_length"])

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        predict_with_generate=True,
        fp16=t["fp16"],
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])
    return cfg["output_dir"]


def load_trocr(model_dir: str) -> "callable":
    """Loads fine-tuned TrOCR from model_dir. Returns extract_text(image) -> OCRResult."""
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()

    def extract_text(image: Image.Image) -> OCRResult:
        import torch
        pixel_values = processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return OCRResult(text=text, confidence=1.0, model="trocr-genealogy")

    return extract_text
