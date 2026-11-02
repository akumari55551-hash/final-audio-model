"""Risk policy tests use explicitly synthetic score sequences, not benchmark data."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from inference.risk_engine import (
    RiskPolicy,
    VoiceRiskEngine,
    calibrate_from_development_csv,
    calibrate_risk_thresholds,
    load_policy_from_config,
    policy_from_config,
    save_calibration,
)

BASE = datetime(2026, 10, 24, tzinfo=UTC)


def policy(**overrides) -> RiskPolicy:
    values = {
        "history_size": 8,
        "minimum_valid_predictions": 3,
        "minimum_usable_windows": 3,
        "alert_threshold": 0.65,
        "high_risk_threshold": 0.85,
        "low_risk_threshold": 0.45,
        "alert_persistence_windows": 2,
        "high_risk_persistence_windows": 2,
        "recovery_persistence_windows": 2,
        "cooldown_seconds": 10.0,
        "correlation_group_seconds": 2.0,
        "ema_alpha": 0.5,
        "threshold_source": "synthetic-development-test",
    }
    values.update(overrides)
    return RiskPolicy(**values)


def prediction(score: float, second: float, *, quality: str = "ok", group: str | None = None):
    return {
        "timestamp_utc": (BASE + timedelta(seconds=second)).isoformat(),
        "bona_fide_score": 1.0 - score,
        "spoof_score": score,
        "audio_quality": {"status": quality, "details": []},
        "correlation_group": group,
    }


def test_insufficient_evidence_until_temporally_distinct_predictions() -> None:
    engine = VoiceRiskEngine(policy())
    first = engine.update(prediction(0.99, 0.0))
    second = engine.update(prediction(0.99, 0.5))
    assert first.alert_level == "insufficient_evidence"
    assert second.alert_level == "insufficient_evidence"
    assert "not_enough_temporally_distinct_windows" in second.reason_codes
    third = engine.update(prediction(0.99, 2.1))
    assert third.alert_level == "high"


def test_overlapping_noisy_input_is_robust_and_not_independent_evidence() -> None:
    engine = VoiceRiskEngine(policy(minimum_usable_windows=2))
    scores = (0.70, 0.95, 0.71, 0.68)
    events = [engine.update(prediction(score, index * 0.5)) for index, score in enumerate(scores)]
    assert all(event.usable_prediction_count == 1 for event in events)
    assert all(event.alert_level == "insufficient_evidence" for event in events)
    later = engine.update(prediction(0.72, 2.1))
    assert later.usable_prediction_count == 2
    assert later.risk_score == pytest.approx(0.71)


def test_alert_persistence_cooldown_and_repeat_notifications() -> None:
    engine = VoiceRiskEngine(policy(alert_persistence_windows=2, high_risk_persistence_windows=2))
    events = [engine.update(prediction(0.92, second)) for second in (0.0, 2.1, 4.2)]
    assert [event.alert_level for event in events[:2]] == ["insufficient_evidence", "insufficient_evidence"]
    assert events[2].alert_level == "high"
    assert events[2].alert_emitted is True

    repeated = engine.update(prediction(0.93, 6.3))
    assert repeated.alert_level == "high"
    assert repeated.alert_emitted is False
    assert "alert_persistence" in repeated.reason_codes

    cooldown_event = engine.update(prediction(0.94, 15.0))
    assert cooldown_event.alert_level == "high"
    assert cooldown_event.alert_emitted is True


def test_recovery_requires_persistent_low_scores_and_rearms() -> None:
    engine = VoiceRiskEngine(policy(recovery_persistence_windows=2))
    for second in (0.0, 2.1, 4.2):
        alert = engine.update(prediction(0.95, second))
    assert alert.alert_level == "high"

    first_low = engine.update(prediction(0.10, 6.3))
    assert first_low.state == "alert_active"
    recovered = engine.update(prediction(0.12, 8.4))
    assert recovered.alert_level == "none"
    assert "risk_recovered_after_persistence" in recovered.reason_codes


def test_missing_or_noisy_audio_is_insufficient_not_low_risk() -> None:
    engine = VoiceRiskEngine(policy())
    missing = engine.mark_missing()
    assert missing.alert_level == "insufficient_evidence"
    assert "prediction_missing" in missing.reason_codes

    bad = engine.update(prediction(0.01, 0, quality="clipping_detected"))
    assert bad.alert_level == "insufficient_evidence"
    assert bad.risk_score is None
    assert "audio_quality_insufficient" in bad.reason_codes


def test_threshold_boundaries_are_inclusive() -> None:
    engine = VoiceRiskEngine(policy(minimum_usable_windows=1, minimum_valid_predictions=1,
                                    alert_persistence_windows=1, high_risk_persistence_windows=1))
    alert = engine.update(prediction(0.65, 0))
    assert alert.alert_level == "elevated"
    assert alert.alert_emitted

    high_engine = VoiceRiskEngine(policy(minimum_usable_windows=1, minimum_valid_predictions=1,
                                         alert_persistence_windows=1, high_risk_persistence_windows=1))
    high = high_engine.update(prediction(0.85, 0))
    assert high.alert_level == "high"


def test_risk_smoothing_ignores_within_group_frame_multiplicity() -> None:
    engine = VoiceRiskEngine(
        policy(
            minimum_usable_windows=1,
            minimum_valid_predictions=1,
            alert_persistence_windows=10,
            high_risk_persistence_windows=10,
        )
    )
    first = engine.update(prediction(0.2, 0.0, group="g1"))
    repeated = [engine.update(prediction(0.9, 0.5, group="g1")) for _ in range(10)]
    assert all(event.usable_prediction_count == 1 for event in repeated)
    assert repeated[-1].risk_score == pytest.approx(first.risk_score)
    assert engine._consecutive_alert_groups == 0


def test_development_threshold_calibration_and_provenance(tmp_path: Path) -> None:
    result = calibrate_risk_thresholds(
        [0, 0, 0, 0, 1, 1, 1],
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9],
        target_false_positive_rate=0.25,
        minimum_alert_threshold=0.4,
        development_source="synthetic-dev-fixture",
        high_risk_margin=0.1,
    )
    assert result.alert_threshold == pytest.approx(0.4)
    assert result.high_risk_threshold == pytest.approx(0.5)
    assert result.low_risk_threshold == pytest.approx(0.3)
    path = tmp_path / "calibration.json"
    save_calibration(path, result)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"risk": yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/baseline.yaml").read_text())["risk"]}),
        encoding="utf-8",
    )
    loaded = policy_from_config(config_path, path)
    assert loaded.alert_threshold == pytest.approx(result.alert_threshold)
    assert loaded.threshold_source == "synthetic-dev-fixture"
    assert load_policy_from_config(config_path).threshold_source == "uncalibrated_placeholder"


def test_calibration_csv_rejects_eval_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("label", "spoof_score", "split"))
        writer.writeheader()
        writer.writerows(
            [
                {"label": 0, "spoof_score": 0.1, "split": "dev"},
                {"label": 1, "spoof_score": 0.8, "split": "eval"},
            ]
        )
    with pytest.raises(ValueError, match="development rows only"):
        calibrate_from_development_csv(path)


def test_policy_rejects_invalid_boundaries_and_bounded_history() -> None:
    with pytest.raises(ValueError, match="Require 0 <= low"):
        RiskPolicy(alert_threshold=0.5, high_risk_threshold=0.9, low_risk_threshold=0.5)
    engine = VoiceRiskEngine(policy(history_size=3, minimum_valid_predictions=1, minimum_usable_windows=1))
    for index in range(7):
        engine.update(prediction(0.2, index * 3.0))
    assert engine.history_size == 3
