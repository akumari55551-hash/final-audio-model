"""Metrics, development-only threshold selection, and small SVG figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    confusion_matrix: list[list[int]]
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float | None
    equal_error_rate: float | None
    support: int
    bona_fide_support: int
    spoof_support: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_scores(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_score.shape:
        raise ValueError("labels and scores must be non-empty and have equal lengths")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must contain only bona_fide=0 and spoof=1")
    if not np.isfinite(y_score).all():
        raise ValueError("scores must all be finite")
    return y_true, y_score


def confusion_from_scores(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> list[list[int]]:
    y_true, y_score = _validate_scores(labels, scores)
    predicted = y_score >= threshold
    tn = int(np.count_nonzero((y_true == 0) & ~predicted))
    fp = int(np.count_nonzero((y_true == 0) & predicted))
    fn = int(np.count_nonzero((y_true == 1) & ~predicted))
    tp = int(np.count_nonzero((y_true == 1) & predicted))
    return [[tn, fp], [fn, tp]]


def select_f1_threshold(
    labels: Sequence[int], spoof_scores: Sequence[float]
) -> tuple[float, dict[str, float]]:
    """Select threshold using development labels only; tie-break conservatively.

    A spoof prediction is score >= threshold. Candidate thresholds include a
    value just above max score (all bona fide) and each observed score. Highest
    spoof F1 wins; ties use the highest threshold, then higher precision.
    """
    y_true, scores = _validate_scores(labels, spoof_scores)
    if not np.any(y_true == 0) or not np.any(y_true == 1):
        raise ValueError("Threshold selection requires both classes in development data")
    candidates = np.unique(np.concatenate(([np.nextafter(scores.max(), np.inf)], scores)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        matrix = confusion_from_scores(y_true, scores, float(threshold))
        _, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, float(threshold), precision, recall)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0]
            and (candidate[1] > best[1] or (candidate[1] == best[1] and candidate[2] > best[2]))
        ):
            best = candidate
    assert best is not None
    return best[1], {"development_spoof_f1": best[0], "development_precision": best[2], "development_recall": best[3]}


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_labels == 1)[boundaries]
    false_positives = 1 + boundaries - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    # Add the end point explicitly; duplicated thresholds may have ended before 1.
    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr = np.r_[tpr, 1.0]
        fpr = np.r_[fpr, 1.0]
    return fpr.astype(np.float64), tpr.astype(np.float64)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y_true, y_score = _validate_scores(labels, scores)
    fpr, tpr = _roc_points(y_true, y_score)
    if not fpr.size:
        return None
    order = np.argsort(fpr, kind="mergesort")
    return float(np.trapezoid(tpr[order], fpr[order]))


def equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Estimate EER from ROC points by linearly interpolating FPR = 1 - TPR."""
    y_true, y_score = _validate_scores(labels, scores)
    fpr, tpr = _roc_points(y_true, y_score)
    if not fpr.size:
        return None
    fnr = 1.0 - tpr
    difference = fpr - fnr
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1.0e-12))
    if exact.size:
        index = int(exact[0])
        return float((fpr[index] + fnr[index]) / 2.0)
    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if crossings.size:
        index = int(crossings[0])
        ratio = -difference[index] / (difference[index + 1] - difference[index])
        interpolated_fpr = fpr[index] + ratio * (fpr[index + 1] - fpr[index])
        interpolated_fnr = fnr[index] + ratio * (fnr[index + 1] - fnr[index])
        return float((interpolated_fpr + interpolated_fnr) / 2.0)
    index = int(np.argmin(np.abs(difference)))
    return float((fpr[index] + fnr[index]) / 2.0)


def compute_binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> BinaryMetrics:
    y_true, y_score = _validate_scores(labels, scores)
    matrix = confusion_from_scores(y_true, y_score, threshold)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return BinaryMetrics(
        threshold=float(threshold),
        confusion_matrix=matrix,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        roc_auc=roc_auc(y_true, y_score),
        equal_error_rate=equal_error_rate(y_true, y_score),
        support=int(y_true.size),
        bona_fide_support=int(np.count_nonzero(y_true == 0)),
        spoof_support=int(np.count_nonzero(y_true == 1)),
    )


def save_roc_svg(
    destination: str,
    series: dict[str, tuple[Sequence[int], Sequence[float]]],
    title: str = "ROC curve",
) -> bool:
    """Write a dependency-free SVG ROC figure; return false if no series is valid."""
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
    plot_left, plot_top, plot_size = 72, 48, 440
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="570" viewBox="0 0 600 570">',
        f'<text x="300" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{_escape(title)}</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_size}" height="{plot_size}" fill="white" stroke="#111827"/>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_size}" x2="{plot_left + plot_size}" y2="{plot_top}" stroke="#9ca3af" stroke-dasharray="6 5"/>',
        '<text x="292" y="535" text-anchor="middle" font-family="sans-serif" font-size="14">False-positive rate</text>',
        '<text x="20" y="270" transform="rotate(-90 20 270)" text-anchor="middle" font-family="sans-serif" font-size="14">True-positive rate</text>',
    ]
    valid_series = 0
    for name, (labels, scores) in series.items():
        y_true, y_score = _validate_scores(labels, scores)
        fpr, tpr = _roc_points(y_true, y_score)
        if not fpr.size:
            continue
        valid_series += 1
        points = " ".join(
            f"{plot_left + f * plot_size:.2f},{plot_top + (1 - t) * plot_size:.2f}"
            for f, t in zip(fpr, tpr)
        )
        color = colors[(valid_series - 1) % len(colors)]
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        elements.append(
            f'<text x="{plot_left + 10}" y="{plot_top + 22 * valid_series}" font-family="sans-serif" font-size="12" fill="{color}">{_escape(name)}</text>'
        )
    if not valid_series:
        return False
    elements.append("</svg>")
    _write_svg(destination, "\n".join(elements))
    return True


def save_confusion_svg(destination: str, matrix: Sequence[Sequence[int]], title: str) -> None:
    cell = 100
    left, top = 160, 75
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="460" height="340" viewBox="0 0 460 340">',
        f'<text x="230" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{_escape(title)}</text>',
        '<text x="280" y="62" text-anchor="middle" font-family="sans-serif" font-size="12">Predicted class</text>',
        '<text x="40" y="180" transform="rotate(-90 40 180)" text-anchor="middle" font-family="sans-serif" font-size="12">True class</text>',
    ]
    labels = ("Bona fide", "Spoof")
    for row in range(2):
        elements.append(f'<text x="{left - 12}" y="{top + row * cell + 56}" text-anchor="end" font-family="sans-serif" font-size="12">{labels[row]}</text>')
        elements.append(f'<text x="{left + row * cell + 50}" y="{top - 10}" text-anchor="middle" font-family="sans-serif" font-size="12">{labels[row]}</text>')
        for column in range(2):
            value = int(matrix[row][column])
            fill = "#dbeafe" if row == column else "#fee2e2"
            x, y = left + column * cell, top + row * cell
            elements.extend(
                (
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>',
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 7}" text-anchor="middle" font-family="sans-serif" font-size="22">{value}</text>',
                )
            )
    elements.append("</svg>")
    _write_svg(destination, "\n".join(elements))


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _write_svg(destination: str, content: str) -> None:
    from pathlib import Path

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
