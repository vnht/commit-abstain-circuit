def compute_confusion(y_true, y_pred):
    """y_true: 1=answerable, y_pred: 1=commit."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def compute_decision_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    return {
        "overall_accuracy":  (tp + tn) / n if n else 0,
        "true_commitment":   tp / (tp + fn) if (tp + fn) else 0,
        "false_abstention":  fn / (tp + fn) if (tp + fn) else 0,
        "true_abstention":   tn / (tn + fp) if (tn + fp) else 0,
        "false_commitment":  fp / (tn + fp) if (tn + fp) else 0,
    }


def compute_auroc(scores, labels):
    """AUROC via Mann-Whitney U statistic. labels: 1=answerable (positive)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    u = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos for n in neg
    )
    return u / (len(pos) * len(neg))
