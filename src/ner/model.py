import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.data.schemas import Entity


def load_model_and_tokenizer(config_path: str = "configs/ner_config.yaml"):
    """Returns (model, tokenizer, id2label) from config — used during training."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    label2id = {label: i for i, label in enumerate(cfg["labels"])}
    id2label = {i: label for i, label in enumerate(cfg["labels"])}

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"],
        num_labels=len(cfg["labels"]),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer, id2label


def load_finetuned(model_dir: str, config_path: str = "configs/ner_config.yaml"):
    """Loads fine-tuned NER model from model_dir — used for inference."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    id2label = {i: label for i, label in enumerate(cfg["labels"])}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model, tokenizer, id2label


def extract_entities(
    text: str,
    model,
    tokenizer,
    id2label: dict,
    max_length: int = 128,
) -> list[Entity]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred_ids = probs.argmax(-1).squeeze().tolist()
    confs = probs.max(-1).values.squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    if isinstance(pred_ids, int):
        pred_ids = [pred_ids]
        confs = [confs]

    entities: list[Entity] = []
    current_label: str | None = None
    current_tokens: list[str] = []
    current_confs: list[float] = []

    for token, pred_id, conf in zip(tokens, pred_ids, confs):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        label = id2label.get(pred_id, "O")

        if label.startswith("B-"):
            if current_label:
                _flush(current_label, current_tokens, current_confs, text, entities)
            current_label = label[2:]
            current_tokens = [token]
            current_confs = [conf]
        elif label.startswith("I-") and current_label == label[2:]:
            current_tokens.append(token)
            current_confs.append(conf)
        else:
            if current_label:
                _flush(current_label, current_tokens, current_confs, text, entities)
            current_label = None
            current_tokens = []
            current_confs = []

    if current_label:
        _flush(current_label, current_tokens, current_confs, text, entities)

    return entities


def _flush(label: str, tokens: list[str], confs: list[float], text: str, entities: list[Entity]) -> None:
    value = _detokenize(tokens)
    idx = text.find(value)
    span = (idx, idx + len(value)) if idx != -1 else (0, len(value))
    avg_conf = sum(confs) / len(confs)
    entities.append(Entity(label=label, value=value, confidence=avg_conf, span=span))


def _detokenize(tokens: list[str]) -> str:
    result = ""
    for token in tokens:
        if token.startswith("##"):
            result += token[2:]
        else:
            result += (" " if result else "") + token
    return result.strip()
