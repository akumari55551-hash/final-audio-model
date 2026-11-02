"""Exact metric tests with synthetic scores; not benchmark measurements."""

from __future__ import annotations

import pytest

from training.evaluation_metrics import (
    compute_binary_metrics,
    equal_error_rate,
    roc_auc,
    select_f1_threshold,
)


def test_perfect_ranking_metrics_and_eer() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)
    assert metrics.false_negative_rate == pytest.approx(0.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.equal_error_rate == pytest.approx(0.0)


def test_auc_and_eer_are_null_without_both_classes() -> None:
    metrics = compute_binary_metrics([1, 1], [0.2, 0.9], threshold=0.5)
    assert metrics.roc_auc is None
    assert metrics.equal_error_rate is None


def test_threshold_selection_uses_labels_and_scores_given_from_dev() -> None:
    threshold, selected = select_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9])
    assert 0.3 < threshold <= 0.6
    assert selected["development_spoof_f1"] == pytest.approx(1.0)


def test_known_auc_ordering() -> None:
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert equal_error_rate([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(0.5)
