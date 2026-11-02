"""Dependency-light classification metrics for bona-fide/spoof training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    precision: float
    recall: float
    f1: float
    spoof_precision: float
    spoof_recall: float
    spoof_f1: float
    confusion_matrix: list[list[int]]
    support: int

    def as_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "precision_macro": self.precision,
            "recall_macro": self.recall,
            "f1_macro": self.f1,
            "spoof_precision": self.spoof_precision,
            "spoof_recall": self.spoof_recall,
            "spoof_f1": self.spoof_f1,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": ["bona_fide_0", "spoof_1"],
            "support": self.support,
        }


def classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    mean_loss: float,
) -> ClassificationMetrics:
    """Compute macro and spoof-positive precision/recall/F1 plus confusion.

    Confusion-matrix rows are true labels and columns are predicted labels,
    ordered as bona fide 0 then spoof 1. Undefined class metrics are set to 0.
    """
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("labels and predictions must be non-empty arrays of equal size")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("labels and predictions must contain only class IDs 0 and 1")
    if not np.isfinite(mean_loss) or mean_loss < 0:
        raise ValueError("mean_loss must be finite and non-negative")

    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    actual_count = confusion.sum(axis=1).astype(np.float64)
    precision_by_class = np.divide(
        true_positive, predicted_count, out=np.zeros(2), where=predicted_count != 0
    )
    recall_by_class = np.divide(true_positive, actual_count, out=np.zeros(2), where=actual_count != 0)
    f1_by_class = np.divide(
        2 * precision_by_class * recall_by_class,
        precision_by_class + recall_by_class,
        out=np.zeros(2),
        where=(precision_by_class + recall_by_class) != 0,
    )
    return ClassificationMetrics(
        loss=float(mean_loss),
        precision=float(precision_by_class.mean()),
        recall=float(recall_by_class.mean()),
        f1=float(f1_by_class.mean()),
        spoof_precision=float(precision_by_class[1]),
        spoof_recall=float(recall_by_class[1]),
        spoof_f1=float(f1_by_class[1]),
        confusion_matrix=confusion.tolist(),
        support=int(y_true.size),
    )
