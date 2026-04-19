import evaluate as hf_evaluate

_seqeval = hf_evaluate.load("seqeval")


def compute_metrics(
    predictions: list[list[str]],
    references: list[list[str]],
) -> dict:
    """
    Returns overall precision, recall, F1, accuracy and per-label breakdown.
    predictions/references: lists of BIO label sequences (one per document).
    """
    results = _seqeval.compute(predictions=predictions, references=references)

    per_label = {
        key: value
        for key, value in results.items()
        if isinstance(value, dict)
    }

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "per_label": per_label,
    }
