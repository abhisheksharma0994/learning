"""The three-line entry point: fit, decide, report.

:class:`CalibratedDecider` bundles what almost every caller wants -- a scorer, a
fitted calibration, and a threshold that honors a target error rate -- without
hiding any of the pieces. It is a convenience over ``TemperatureScaler`` plus
``ConformalRouter``, not a replacement for them: if you need the per-label view
or a different calibrator, use those directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .calibrate import HistogramBinner, TemperatureScaler
from .decide import ConformalRouter, Decision, select
from .metrics import argmax, ece, nll, top1_accuracy

__all__ = ["CalibratedDecider", "CalibrationReport"]


@dataclass(frozen=True)
class CalibrationReport:
    """Held-out numbers for one decider. Every field is measured, none inferred."""

    n_cal: int
    n_test: int
    accuracy: float
    raw_ece: float
    calibrated_ece: float
    nll: float
    threshold: float | None
    coverage: float
    risk: float

    def lines(self) -> list[str]:
        threshold = "unreachable" if self.threshold is None else f"{self.threshold:.3f}"
        risk = "n/a" if self.risk != self.risk else f"{self.risk:.1%}"
        return [
            f"labelled decisions        {self.n_cal} calibration / {self.n_test} held out",
            f"accuracy on held out      {self.accuracy:.3f}",
            f"ECE raw -> calibrated     {self.raw_ece:.4f} -> {self.calibrated_ece:.4f}",
            f"nll                       {self.nll:.4f}",
            f"accept if confidence >=   {threshold}",
            f"coverage / risk           {self.coverage:.1%} auto-decided at {risk} error",
        ]

    def __str__(self) -> str:
        return "\n".join(self.lines())


class CalibratedDecider:
    """A scorer plus the calibration and threshold that make it usable.

    Both calibrators here are monotone, so no decision is changed by fitting --
    only the numbers attached to decisions. What changes is that a threshold
    starts to mean something.
    """

    def __init__(
        self,
        scorer,
        calibrator: TemperatureScaler | HistogramBinner,
        router: ConformalRouter,
        label_set,
        n_cal: int,
    ) -> None:
        self.scorer = scorer
        self.calibrator = calibrator
        self.router = router
        self.label_set = label_set
        self.n_cal = n_cal

    @classmethod
    def fit(
        cls,
        scorer,
        cal_states: Sequence[str],
        cal_labels: Sequence[int],
        *,
        target_risk: float = 0.05,
        method: str = "temperature",
        n_bins: int = 15,
    ) -> "CalibratedDecider":
        """Fit on labelled decisions. ``scorer`` must expose ``label_set`` and ``score``."""
        if len(cal_states) != len(cal_labels):
            raise ValueError("cal_states and cal_labels must be the same length")
        if not cal_states:
            raise ValueError("need at least one calibration example")
        if method not in ("temperature", "histogram"):
            raise ValueError("method must be 'temperature' or 'histogram'")

        label_set = getattr(scorer, "label_set", None)
        if label_set is None:
            raise ValueError("scorer must expose a label_set")

        raw = scorer.score(cal_states)
        calibrator = (
            TemperatureScaler().fit(raw, cal_labels)
            if method == "temperature"
            else HistogramBinner(n_bins=n_bins).fit(raw, cal_labels)
        )
        calibrated = calibrator.transform(raw)
        confidences = [max(row) for row in calibrated]
        correct = [argmax(row) == target for row, target in zip(calibrated, cal_labels)]
        router = ConformalRouter(target_risk=target_risk).fit(confidences, correct)

        return cls(
            scorer=scorer,
            calibrator=calibrator,
            router=router,
            label_set=label_set,
            n_cal=len(cal_states),
        )

    # ----------------------------------------------------------------- usage

    def probabilities(self, state: str) -> list[float]:
        """Calibrated distribution over the label set for one state."""
        return self.calibrator.transform(self.scorer.score([state]))[0]

    def decide(self, state: str) -> tuple[Decision, list[float], bool]:
        """Return ``(decision, calibrated probabilities, accepted)``.

        ``accepted`` is False when the confidence falls below the fitted
        threshold, which means: escalate, ask, or retry -- do not ship it.
        """
        probs = self.probabilities(state)
        decision = select(probs, self.label_set)
        accepted = self.router.route([decision.confidence])[0]
        return decision, probs, accepted

    def evaluate(self, test_states: Sequence[str], test_labels: Sequence[int]) -> CalibrationReport:
        """Measure on data the calibration never saw."""
        if len(test_states) != len(test_labels):
            raise ValueError("test_states and test_labels must be the same length")
        raw = self.scorer.score(test_states)
        calibrated = self.calibrator.transform(raw)
        risk, coverage = self.router.evaluate(
            [max(row) for row in calibrated],
            [argmax(row) == target for row, target in zip(calibrated, test_labels)],
        )
        return CalibrationReport(
            n_cal=self.n_cal,
            n_test=len(test_states),
            accuracy=top1_accuracy(calibrated, test_labels),
            raw_ece=ece(raw, test_labels),
            calibrated_ece=ece(calibrated, test_labels),
            nll=nll(calibrated, test_labels),
            threshold=self.router.threshold_ if self.router.reachable else None,
            coverage=coverage,
            risk=risk,
        )

    def __repr__(self) -> str:
        return (
            f"CalibratedDecider(labels={len(self.label_set)}, "
            f"target_risk={self.router.target_risk:.3f}, "
            f"threshold={self.router.threshold_})"
        )
