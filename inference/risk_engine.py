"""Conservative continuous spoof-risk assessment for model estimates.

Scores are not probabilities of fraud or identity. This policy layer smooths
correlated overlapping windows, enforces evidence/persistence rules, and emits
human-review guidance only; it cannot approve transactions, deny service, or
verify caller identity.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class RiskPolicy:
    """Explicit operating policy; thresholds must be calibrated on dev data."""

    history_size: int = 12
    minimum_valid_predictions: int = 3
    minimum_usable_windows: int = 3
    alert_threshold: float = 0.80
    high_risk_threshold: float = 0.95
    low_risk_threshold: float = 0.50
    alert_persistence_windows: int = 2
    high_risk_persistence_windows: int = 3
    recovery_persistence_windows: int = 3
    cooldown_seconds: float = 15.0
    smoothing_method: str = "median_then_ema"
    ema_alpha: float = 0.35
    correlation_group_seconds: float = 2.0
    minimum_audio_quality: str = "ok"
    calibration_path: str | None = None
    threshold_source: str = "development_calibrated"

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if not 1 <= self.minimum_valid_predictions <= self.history_size:
            raise ValueError("minimum_valid_predictions must be within bounded history")
        if self.minimum_usable_windows < 1:
            raise ValueError("minimum_usable_windows must be positive")
        if not 0.0 <= self.low_risk_threshold < self.alert_threshold < self.high_risk_threshold <= 1.0:
            raise ValueError("Require 0 <= low < alert < high <= 1")
        if min(self.alert_persistence_windows, self.high_risk_persistence_windows, self.recovery_persistence_windows) < 1:
            raise ValueError("Persistence window counts must be positive")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.correlation_group_seconds <= 0.0:
            raise ValueError("correlation_group_seconds must be positive")
        if self.smoothing_method not in {"median_then_ema", "median_only"}:
            raise ValueError("smoothing_method must be median_then_ema or median_only")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> RiskPolicy:
        accepted = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in accepted})


@dataclass(frozen=True)
class PredictionObservation:
    timestamp: datetime
    bona_fide_score: float
    spoof_score: float
    audio_quality: dict[str, Any]
    inference_latency_ms: float | None = None
    sequence: int | None = None
    correlation_group: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PredictionObservation:
        score = float(value["spoof_score"])
        bona_fide = float(value.get("bona_fide_score", 1.0 - score))
        timestamp_value = value.get("timestamp_utc")
        if isinstance(timestamp_value, datetime):
            timestamp = timestamp_value
        elif timestamp_value is None:
            timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return cls(
            timestamp=timestamp,
            bona_fide_score=bona_fide,
            spoof_score=score,
            audio_quality=value.get("audio_quality") or {},
            inference_latency_ms=(
                float(value["inference_latency_ms"]) if value.get("inference_latency_ms") is not None else None
            ),
            sequence=int(value["sequence"]) if value.get("sequence") is not None else None,
            correlation_group=(str(value["correlation_group"]) if value.get("correlation_group") is not None else None),
        )


@dataclass(frozen=True)
class RiskEvent:
    timestamp: str
    state: str
    risk_score: float | None
    model_confidence: dict[str, Any] | None
    alert_emitted: bool
    alert_level: str
    reason_codes: tuple[str, ...]
    recommended_verification_actions: tuple[str, ...]
    usable_prediction_count: int
    history_size: int
    threshold_source: str
    correlation_note: str
    automated_action: str = "none"
    identity_statement: str = "Not verified identity; model output is an uncertain signal only."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationResult:
    alert_threshold: float
    high_risk_threshold: float
    target_false_positive_rate: float
    development_examples: int
    development_source: str
    method: str
    low_risk_threshold: float = 0.0
    score_convention: str = "spoof score from class index 1; threshold labels are selected on development data only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not y.size or y.shape != s.shape:
        raise ValueError("development labels and spoof scores must be non-empty and equally sized")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be bona_fide=0 or spoof=1")
    if not np.isfinite(s).all() or np.any((s < 0.0) | (s > 1.0)):
        raise ValueError("spoof scores must be finite values in [0, 1]")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("threshold calibration requires both classes in development data")
    return y, s


def calibrate_risk_thresholds(
    development_labels: Sequence[int],
    development_spoof_scores: Sequence[float],
    *,
    target_false_positive_rate: float = 0.01,
    development_source: str = "development",
    minimum_alert_threshold: float = 0.50,
    high_risk_margin: float = 0.10,
) -> CalibrationResult:
    """Choose alert threshold by dev bona-fide FPR; reserve a higher tier.

    Selects the lowest observed dev score threshold that achieves FPR <= the
    requested target (thus maximizing detection under that FPR constraint).
    If ties leave no room for a strict high threshold, calibration fails. This
    is a threshold transfer rule, not a calibration of model scores into fraud
    probabilities and not a guarantee of the same FPR in deployment.
    """
    y, scores = _score_arrays(development_labels, development_spoof_scores)
    if not 0.0 <= target_false_positive_rate < 1.0:
        raise ValueError("target_false_positive_rate must be in [0, 1)")
    if not 0.0 <= minimum_alert_threshold < 1.0:
        raise ValueError("minimum_alert_threshold must be in [0, 1)")
    if not 0.0 < high_risk_margin <= 1.0:
        raise ValueError("high_risk_margin must be in (0, 1]")
    bona_fide_scores = np.sort(scores[y == 0])
    candidates = np.unique(np.concatenate(([minimum_alert_threshold], scores)))
    feasible = [
        float(threshold)
        for threshold in candidates
        if threshold >= minimum_alert_threshold
        and np.mean(bona_fide_scores >= threshold) <= target_false_positive_rate
    ]
    if not feasible:
        raise ValueError("Could not find a development threshold meeting the requested false-positive target")
    alert_threshold = min(feasible)
    if alert_threshold >= 1.0:
        raise ValueError("Development data provides no usable alert threshold below 1.0")
    high_threshold = min(1.0, alert_threshold + high_risk_margin)
    if high_threshold <= alert_threshold:
        raise ValueError("No distinct high-risk threshold can be formed")
    low_threshold = max(0.0, alert_threshold - high_risk_margin)
    return CalibrationResult(
        alert_threshold=alert_threshold,
        high_risk_threshold=high_threshold,
        low_risk_threshold=low_threshold,
        target_false_positive_rate=target_false_positive_rate,
        development_examples=int(y.size),
        development_source=development_source,
        method="lowest candidate score meeting target development bona-fide FPR; high threshold adds configured margin",
    )


def save_calibration(path: Path | str, result: CalibrationResult) -> None:
    """Persist development calibration provenance and thresholds as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_policy_from_config(config_path: Path | str) -> RiskPolicy:
    """Read the `risk` mapping from the shared project YAML configuration."""
    path = Path(config_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load risk policy from {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("risk"), dict):
        raise TypeError("Configuration must contain a risk mapping")
    return RiskPolicy.from_mapping(raw["risk"])


class VoiceRiskEngine:
    """Bounded, persistence-aware risk-state engine for overlapping windows.

    Correlation handling has two levels: adjacent windows whose timestamps fall
    within ``correlation_group_seconds`` are reduced to a median score per
    temporal group, and the final risk score is a median over distinct groups.
    Thus repeated highly overlapping frames are not counted as independent
    evidence. A one-window EMA is optionally maintained for display smoothing;
    evidence counts and persistence operate on temporal groups, not windows.

    Alert transitions have cooldowns and re-arm only after sustained low risk.
    Inadequate/poor quality input yields ``insufficient_evidence`` and never
    maps to a low-risk/allow decision.
    """

    def __init__(self, policy: RiskPolicy | None = None, *, calibration: CalibrationResult | None = None):
        self.policy = policy or RiskPolicy()
        if calibration is not None:
            self.policy = RiskPolicy(
                **{
                    **asdict(self.policy),
                    "alert_threshold": calibration.alert_threshold,
                    "high_risk_threshold": calibration.high_risk_threshold,
                    "threshold_source": calibration.development_source,
                }
            )
        self._history: deque[PredictionObservation] = deque(maxlen=self.policy.history_size)
        self._state = "insufficient_evidence"
        self._active_alert = "none"
        self._last_alert_time: datetime | None = None
        self._last_risk_score: float | None = None
        self._ema_score: float | None = None
        self._consecutive_alert_groups = 0
        self._consecutive_high_groups = 0
        self._consecutive_low_groups = 0
        self._last_group_key: str | None = None
        self._last_group_timestamp: datetime | None = None
        self._current_group_start: datetime | None = None
        self._usable_groups: deque[tuple[str, datetime, float]] = deque(maxlen=self.policy.history_size)
        self._current_group_key: str | None = None
        self._current_group_scores: deque[float] = deque(maxlen=self.policy.history_size)
        self._last_processed_group_key: str | None = None

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def state(self) -> str:
        return self._state

    @property
    def policy(self) -> RiskPolicy:
        return self._policy

    @policy.setter
    def policy(self, value: RiskPolicy) -> None:
        self._policy = value

    def reset(self) -> None:
        """Clear temporal evidence and alert state for a new call/session."""
        self._history.clear()
        self._usable_groups.clear()
        self._state = "insufficient_evidence"
        self._active_alert = "none"
        self._last_alert_time = None
        self._last_risk_score = None
        self._ema_score = None
        self._consecutive_alert_groups = 0
        self._consecutive_high_groups = 0
        self._consecutive_low_groups = 0
        self._last_group_key = None
        self._last_group_timestamp = None
        self._current_group_start = None
        self._current_group_key = None
        self._current_group_scores.clear()
        self._last_processed_group_key = None

    def mark_missing(self, *, timestamp: datetime | None = None, reason: str = "prediction_missing") -> RiskEvent:
        """Handle no model output as insufficient evidence, never as safe."""
        now = timestamp or datetime.now(UTC)
        self._state = "insufficient_evidence"
        self._consecutive_low_groups = 0
        return self._event(
            now,
            score=None,
            level="insufficient_evidence",
            reasons=(reason,),
            actions=self._verification_actions("insufficient_evidence"),
            confidence=None,
        )

    def update(self, prediction: PredictionObservation | dict[str, Any] | None) -> RiskEvent:
        """Consume a model window prediction and produce a structured risk event."""
        if prediction is None:
            return self.mark_missing()
        observation = (
            prediction if isinstance(prediction, PredictionObservation) else PredictionObservation.from_mapping(prediction)
        )
        if observation.timestamp.tzinfo is None:
            observation = PredictionObservation(
                timestamp=observation.timestamp.replace(tzinfo=UTC),
                bona_fide_score=observation.bona_fide_score,
                spoof_score=observation.spoof_score,
                audio_quality=observation.audio_quality,
                inference_latency_ms=observation.inference_latency_ms,
                sequence=observation.sequence,
                correlation_group=observation.correlation_group,
            )
        self._history.append(observation)
        quality_status = str(observation.audio_quality.get("status", "unknown")).casefold()
        reasons: list[str] = []
        if quality_status != self.policy.minimum_audio_quality:
            reasons.append("audio_quality_insufficient")
        if observation.audio_quality.get("dropped_audio_chunks", 0) or observation.audio_quality.get(
            "dropped_audio_samples", 0
        ):
            reasons.append("audio_capture_discontinuity")
        if observation.audio_quality.get("skipped_inference_windows", 0):
            reasons.append("inference_backlog")
        if observation.audio_quality.get("details"):
            details = observation.audio_quality["details"]
            if any("silence" in str(detail) for detail in details):
                reasons.append("silence_or_low_level_audio")
            if any("clipping" in str(detail) for detail in details):
                reasons.append("clipped_audio")

        valid_scores = (
            math.isfinite(observation.spoof_score)
            and math.isfinite(observation.bona_fide_score)
            and 0.0 <= observation.spoof_score <= 1.0
            and 0.0 <= observation.bona_fide_score <= 1.0
        )
        if not valid_scores:
            reasons.append("invalid_model_score")
        if reasons:
            self._state = "insufficient_evidence"
            self._consecutive_low_groups = 0
            return self._event(
                observation.timestamp,
                score=None,
                level="insufficient_evidence",
                reasons=tuple(dict.fromkeys(reasons)),
                actions=self._verification_actions("insufficient_evidence"),
                confidence=None,
            )

        group_key = self._correlation_group(observation)
        new_group = self._add_to_group(observation, group_key)
        median_current = self._current_group_median(group_key)
        # Smooth one median per temporal group; correlated frames never advance
        # the EMA or persistence counters as independent observations.
        group_scores = [score for _, _, score in self._usable_groups]
        median_distinct_groups = float(statistics.median(group_scores)) if group_scores else median_current
        risk_score = (
            float(self._ema_score if self._ema_score is not None else median_current)
            if self.policy.smoothing_method == "median_then_ema"
            else median_distinct_groups
        )
        self._last_risk_score = risk_score
        distinct_groups = len(self._usable_groups)
        alert = risk_score >= self.policy.alert_threshold
        high = risk_score >= self.policy.high_risk_threshold
        low = risk_score < self.policy.low_risk_threshold
        if new_group:
            self._consecutive_alert_groups = self._consecutive_alert_groups + 1 if alert else 0
            self._consecutive_high_groups = self._consecutive_high_groups + 1 if high else 0
            self._consecutive_low_groups = self._consecutive_low_groups + 1 if low else 0
            self._last_processed_group_key = group_key
        valid_count = sum(
            1
            for item in self._history
            if str(item.audio_quality.get("status", "unknown")).casefold()
            == self.policy.minimum_audio_quality
            and math.isfinite(item.spoof_score)
            and 0.0 <= item.spoof_score <= 1.0
        )
        if (
            distinct_groups < self.policy.minimum_usable_windows
            or valid_count < self.policy.minimum_valid_predictions
        ):
            self._state = "insufficient_evidence"
            if distinct_groups < self.policy.minimum_usable_windows:
                reasons.append("not_enough_temporally_distinct_windows")
            if valid_count < self.policy.minimum_valid_predictions:
                reasons.append("not_enough_valid_predictions")
            return self._event(
                observation.timestamp,
                score=risk_score,
                level="insufficient_evidence",
                reasons=tuple(reasons),
                actions=self._verification_actions("insufficient_evidence"),
                confidence=self._model_confidence(risk_score),
            )

        if self._active_alert != "none":
            if self._consecutive_low_groups >= self.policy.recovery_persistence_windows:
                self._active_alert = "none"
                self._state = "monitoring"
                self._consecutive_alert_groups = 0
                self._consecutive_high_groups = 0
                reasons.append("risk_recovered_after_persistence")
            else:
                self._state = "alert_active"
                reasons.append("alert_persistence")
                notify = (
                    self._last_alert_time is None
                    or (observation.timestamp - self._last_alert_time).total_seconds()
                    >= self.policy.cooldown_seconds
                )
                if notify:
                    self._last_alert_time = observation.timestamp
                return self._event(
                    observation.timestamp,
                    score=risk_score,
                    level=self._active_alert,
                    reasons=tuple(reasons or ["elevated_spoof_score"]),
                    actions=self._verification_actions(self._active_alert),
                    confidence=self._model_confidence(risk_score),
                    alert_emitted=notify,
                )

        cooldown_elapsed = self._last_alert_time is None or (
            observation.timestamp - self._last_alert_time
        ).total_seconds() >= self.policy.cooldown_seconds
        if high and self._consecutive_high_groups >= self.policy.high_risk_persistence_windows:
            if cooldown_elapsed:
                self._active_alert = "high"
                self._last_alert_time = observation.timestamp
                self._state = "alert_active"
                reasons.append("high_risk_persistence_met")
                level = "high"
                notify = True
            else:
                self._state = "cooldown"
                reasons.append("high_risk_threshold_met_during_cooldown")
                level = "elevated"
                notify = False
        elif alert and self._consecutive_alert_groups >= self.policy.alert_persistence_windows:
            if cooldown_elapsed:
                self._active_alert = "elevated"
                self._last_alert_time = observation.timestamp
                self._state = "alert_active"
                reasons.append("alert_persistence_met")
                level = "elevated"
                notify = True
            else:
                self._state = "cooldown"
                reasons.append("alert_threshold_met_during_cooldown")
                level = "elevated"
                notify = False
        else:
            self._state = "monitoring"
            level = "none"
            notify = False
            reasons.append("below_alert_threshold" if not alert else "persistence_not_met")

        return self._event(
            observation.timestamp,
            score=risk_score,
            level=level,
            reasons=tuple(reasons),
            actions=self._verification_actions(level),
            confidence=self._model_confidence(risk_score),
            alert_emitted=notify,
        )

    def _correlation_group(self, observation: PredictionObservation) -> str:
        if observation.correlation_group:
            return observation.correlation_group
        if self._current_group_start is None or (
            observation.timestamp - self._current_group_start
        ).total_seconds() >= self.policy.correlation_group_seconds:
            return f"time:{observation.timestamp.timestamp():.6f}"
        return self._last_group_key or f"time:{observation.timestamp.timestamp():.6f}"

    def _add_to_group(self, observation: PredictionObservation, group_key: str) -> bool:
        new_group = group_key != self._current_group_key
        if group_key == self._current_group_key:
            self._current_group_scores.append(observation.spoof_score)
            if self._usable_groups and self._usable_groups[-1][0] == group_key:
                _, group_time, _ = self._usable_groups.pop()
                self._usable_groups.append(
                    (group_key, group_time, float(statistics.median(self._current_group_scores)))
                )
        else:
            if self._current_group_key is not None and self._current_group_scores:
                finalized_score = float(statistics.median(self._current_group_scores))
                if self._ema_score is None:
                    self._ema_score = finalized_score
                else:
                    alpha = self.policy.ema_alpha
                    self._ema_score = alpha * finalized_score + (1.0 - alpha) * self._ema_score
            elif self._ema_score is None:
                # Seed display smoothing from the first group. Alerting is still
                # gated by the distinct-group and valid-record minimums.
                self._ema_score = observation.spoof_score
            self._current_group_key = group_key
            self._current_group_start = observation.timestamp
            self._current_group_scores = deque([observation.spoof_score], maxlen=self.policy.history_size)
            self._usable_groups.append((group_key, observation.timestamp, observation.spoof_score))
        self._last_group_key = group_key
        self._last_group_timestamp = observation.timestamp
        return new_group

    def _current_group_median(self, group_key: str) -> float:
        if self._usable_groups and self._usable_groups[-1][0] == group_key:
            return self._usable_groups[-1][2]
        return float(self._history[-1].spoof_score)

    def _model_confidence(self, risk_score: float) -> dict[str, Any]:
        # These are classifier score summaries, explicitly not real-world fraud probabilities.
        return {
            "bona_fide_model_score": 1.0 - risk_score,
            "spoof_model_score": risk_score,
            "meaning": "smoothed classifier output; not probability of fraud or verified identity",
        }

    @staticmethod
    def _verification_actions(level: str) -> tuple[str, ...]:
        if level == "high":
            return (
                "Pause high-impact action for human review; do not automatically reject or approve.",
                "Verify through a separately established, trusted contact channel.",
                "Use a challenge-response or organizational callback procedure.",
                "Do not rely on caller ID or contextual claims as proof of authenticity.",
            )
        if level == "elevated":
            return (
                "Request independent verification using a trusted channel before proceeding.",
                "Apply the organization's human review procedure for sensitive requests.",
                "Do not rely on caller ID or contextual claims as proof of authenticity.",
            )
        if level == "insufficient_evidence":
            return (
                "Do not treat missing/poor-quality audio as bona fide or spoof evidence.",
                "Improve the audio connection or request a fresh sample with consent.",
                "Use an independent verification procedure for sensitive requests.",
            )
        return (
            "Continue normal independent identity verification for sensitive requests.",
            "A low model score does not verify identity or authorize a transaction.",
        )

    def _event(
        self,
        timestamp: datetime,
        *,
        score: float | None,
        level: str,
        reasons: tuple[str, ...],
        actions: tuple[str, ...],
        confidence: dict[str, Any] | None,
        alert_emitted: bool = False,
    ) -> RiskEvent:
        return RiskEvent(
            timestamp=timestamp.astimezone(UTC).isoformat(),
            state=self._state,
            risk_score=score,
            model_confidence=confidence,
            alert_emitted=alert_emitted,
            alert_level=level,
            reason_codes=reasons,
            recommended_verification_actions=actions,
            usable_prediction_count=len(self._usable_groups),
            history_size=len(self._history),
            threshold_source=self.policy.threshold_source,
            correlation_note=(
                "Overlapping windows within the configured time group are median-aggregated; "
                "the risk score is the median of distinct groups, not independent evidence."
            ),
        )


def calibrate_from_development_csv(
    csv_path: Path | str,
    *,
    score_column: str = "spoof_score",
    label_column: str = "label",
    target_false_positive_rate: float = 0.01,
    development_source: str = "development",
    minimum_alert_threshold: float = 0.50,
    high_risk_margin: float = 0.10,
) -> CalibrationResult:
    """Read a labeled development prediction CSV; reject non-dev split rows."""
    import csv

    path = Path(csv_path)
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {label_column, score_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Development CSV must contain columns {sorted(required)}")
        if "split" not in reader.fieldnames:
            raise ValueError("Development calibration CSV must include a split column for provenance")
        for line, row in enumerate(reader, start=2):
            if row["split"].strip().casefold() not in {"dev", "development"}:
                raise ValueError(f"{path}:{line}: threshold calibration accepts development rows only")
            labels.append(int(row[label_column]))
            scores.append(float(row[score_column]))
    return calibrate_risk_thresholds(
        labels,
        scores,
        target_false_positive_rate=target_false_positive_rate,
        development_source=development_source,
        minimum_alert_threshold=minimum_alert_threshold,
        high_risk_margin=high_risk_margin,
    )


def policy_from_config(config_path: Path | str, calibration_path: Path | str | None = None) -> RiskPolicy:
    """Load policy YAML; optional dev calibration replaces threshold values."""
    policy = load_policy_from_config(config_path)
    if calibration_path is None:
        return policy
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    result = CalibrationResult(**calibration)
    return RiskPolicy(
        **{
            **asdict(policy),
            "alert_threshold": result.alert_threshold,
            "high_risk_threshold": result.high_risk_threshold,
            "low_risk_threshold": result.low_risk_threshold,
            "calibration_path": str(Path(calibration_path).resolve()),
            "threshold_source": result.development_source,
        }
    )
