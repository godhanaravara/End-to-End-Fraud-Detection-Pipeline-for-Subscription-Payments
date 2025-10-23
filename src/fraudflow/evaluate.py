from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def _prec_recall_at_frac(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> tuple[float, float]:
    n = len(y_score)
    k = max(1, int(n * frac))
    idx = np.argpartition(-y_score, k-1)[:k]
    y_top = y_true[idx]
    precision = float(y_top.mean())
    recall = float(y_top.sum() / max(y_true.sum(), 1))
    return precision, recall

def full_metrics(y_true: pd.Series, y_score: pd.Series) -> dict:
    yt = y_true.to_numpy()
    ys = y_score.to_numpy()
    p001, r001 = _prec_recall_at_frac(yt, ys, 0.001)  # 0.1%
    p005, r005 = _prec_recall_at_frac(yt, ys, 0.005)  # 0.5%
    p010, r010 = _prec_recall_at_frac(yt, ys, 0.010)  # 1.0%
    return {
        "roc_auc": float(roc_auc_score(yt, ys)),
        "pr_auc": float(average_precision_score(yt, ys)),
        "precision_at_0.1pct": p001, "recall_at_0.1pct": r001,
        "precision_at_0.5pct": p005, "recall_at_0.5pct": r005,
        "precision_at_1pct":   p010, "recall_at_1pct":   r010,
    }
