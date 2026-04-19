from datasets import load_dataset
from PIL import Image
from src.data.schemas import Document


def load_iam(max_train: int = 2000, max_val: int = 500) -> tuple[list[Document], list[Document]]:
    """Loads IAM Handwriting Database line-level samples for OCR fine-tuning."""
    ds = load_dataset("Teklia/IAM-line", trust_remote_code=True)

    def _to_doc(sample) -> Document:
        return Document(
            image=sample["image"].convert("RGB"),
            text=sample["text"],
            entities=[],
            source="iam",
        )

    train_size = min(max_train, len(ds["train"]))
    val_size = min(max_val, len(ds["validation"]))
    train = [_to_doc(s) for s in ds["train"].select(range(train_size))]
    val = [_to_doc(s) for s in ds["validation"].select(range(val_size))]
    return train, val


def load_funsd() -> tuple[list[Document], list[Document]]:
    """Loads FUNSD scanned form dataset for OCR fine-tuning (image→concatenated text)."""
    ds = load_dataset("nielsr/funsd", trust_remote_code=True)

    def _to_doc(sample) -> Document:
        words = [w for w in sample["words"] if w.strip()]
        text = " ".join(words)
        return Document(
            image=sample["image"].convert("RGB"),
            text=text,
            entities=[],
            source="funsd",
        )

    train = [_to_doc(s) for s in ds["train"]]
    test = [_to_doc(s) for s in ds["test"]]
    return train, test
