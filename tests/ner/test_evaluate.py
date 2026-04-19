from src.ner.evaluate import compute_metrics


def test_perfect_prediction_gives_perfect_metrics():
    preds = [["B-NAME", "I-NAME", "B-DATE"]]
    refs = [["B-NAME", "I-NAME", "B-DATE"]]
    metrics = compute_metrics(preds, refs)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_wrong_prediction_degrades_metrics():
    preds = [["B-NAME", "O", "B-DATE"]]
    refs = [["B-NAME", "I-NAME", "B-DATE"]]
    metrics = compute_metrics(preds, refs)
    assert metrics["f1"] < 1.0


def test_per_label_metrics_included():
    preds = [["B-NAME", "I-NAME"]]
    refs = [["B-NAME", "I-NAME"]]
    metrics = compute_metrics(preds, refs)
    assert "per_label" in metrics
    assert "NAME" in metrics["per_label"]
