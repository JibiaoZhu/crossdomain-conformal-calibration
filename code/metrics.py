import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(scores, labels):
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auprc(scores, labels):
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, scores))


def preds_range_f1(preds, labels):
    preds = np.asarray(preds, dtype=bool)
    tru = np.asarray(labels).astype(bool)
    tp = int(np.sum(preds & tru))
    if tp == 0 or preds.sum() == 0 or tru.sum() == 0:
        return {"range_precision": 0.0, "range_recall": 0.0, "range_f1": 0.0}
    precision = tp / int(preds.sum())
    recall = tp / int(tru.sum())
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"range_precision": float(precision), "range_recall": float(recall), "range_f1": float(f1)}


def fpr_on_normal(preds, normal_mask):
    preds = np.asarray(preds, dtype=bool)
    normal_mask = np.asarray(normal_mask, dtype=bool)
    if not normal_mask.any():
        return float("nan")
    return float(preds[normal_mask].mean())


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    n = x.size
    from scipy import stats
    t = r * np.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))
    return r, p
