import yaml
import numpy as np
import evaluate as hf_evaluate
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from src.data.schemas import Document
from src.ner.model import load_model_and_tokenizer

_seqeval = hf_evaluate.load("seqeval")


class _NERDataset(Dataset):
    def __init__(self, documents: list[Document], tokenizer, label2id: dict, max_length: int):
        self.documents = documents
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict:
        doc = self.documents[idx]
        words = doc.text.split()
        word_labels = _align_labels(words, doc)
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        labels = []
        prev_word_id = None
        for word_id in encoding.word_ids():
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(self.label2id.get(word_labels[word_id], 0))
            else:
                labels.append(-100)
            prev_word_id = word_id
        encoding["labels"] = labels
        return dict(encoding)


def _align_labels(words: list[str], doc: Document) -> list[str]:
    word_labels = ["O"] * len(words)
    for entity in doc.entities:
        entity_words = entity.value.split()
        for i in range(len(words) - len(entity_words) + 1):
            if words[i: i + len(entity_words)] == entity_words:
                word_labels[i] = f"B-{entity.label}"
                for j in range(1, len(entity_words)):
                    word_labels[i + j] = f"I-{entity.label}"
                break
    return word_labels


def finetune(
    train_docs: list[Document],
    val_docs: list[Document],
    config_path: str = "configs/ner_config.yaml",
) -> str:
    """Fine-tunes BERT-NER on train_docs, evaluates on val_docs. Returns output_dir."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer, id2label = load_model_and_tokenizer(config_path)
    label2id = {v: k for k, v in id2label.items()}
    t = cfg["training"]

    train_ds = _NERDataset(train_docs, tokenizer, label2id, t["max_seq_length"])
    val_ds = _NERDataset(val_docs, tokenizer, label2id, t["max_seq_length"])

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_preds = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for l in label_row if l != -100]
            for label_row in labels
        ]
        result = _seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        fp16=t["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    return cfg["output_dir"]
