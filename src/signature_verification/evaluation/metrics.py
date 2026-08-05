from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "threshold": float(thresholds[index]),
        "eer": float((fpr[index] + fnr[index]) / 2),
        "far": float(fpr[index]),
        "frr": float(fnr[index]),
    }


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    negative_types: list[str] | None = None,
) -> dict:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    genuine = labels == 1
    negative = labels == 0
    metrics = {
        "threshold": float(threshold),
        "far": float(np.mean(predictions[negative] == 1)),
        "frr": float(np.mean(predictions[genuine] == 0)),
        "gar": float(np.mean(predictions[genuine] == 1)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "samples": len(labels),
    }
    if negative_types is not None:
        kinds = np.asarray(negative_types)
        for kind in ("SKILLED", "RANDOM"):
            mask = negative & (kinds == kind)
            if mask.any():
                metrics[f"{kind.lower()}_forgery_accuracy"] = float(
                    np.mean(predictions[mask] == 0)
                )
    return metrics
