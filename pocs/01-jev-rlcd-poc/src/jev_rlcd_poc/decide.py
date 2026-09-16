"""Turning probabilities into actions, including the option to abstain.

The point of calibration is not prettier numbers -- it is that a threshold on
confidence becomes meaningful. If the model says 0.9 and is right 90% of the
time, you can ship decisions at 0.9 and route the rest to a human. If the model
says 0.9 and is right 60% of the time, no threshold gives you the error rate you
asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .metrics import argmax, confidence, selective_risk

__all__ = ["ConformalRouter", "Decision", "entropy", "margin", "select"]


@dataclass(frozen=True)
class Decision:
    """One selected outcome, with the evidence behind it."""

    label: str
    index: int
    confidence: float
    probs: tuple[float, ...]
    action: str = "accept"

    @property
    def abstained(self) -> bool:
        return self.action != "accept"


def select(probs: Sequence[float], label_set) -> Decision:
    """Pick the argmax label and carry its confidence along."""
    index = argmax(probs)
    return Decision(
        label=label_set[index],
        index=index,
        confidence=confidence(probs),
        probs=tuple(probs),
    )


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def normalized_entropy(probs: Sequence[float]) -> float:
    """Entropy scaled to [0, 1], so thresholds transfer across label-set sizes."""
    if len(probs) < 2:
        return 0.0
    return entropy(probs) / math.log(len(probs))


def margin(probs: Sequence[float]) -> float:
    """Top probability minus the runner-up. Small means genuinely ambiguous."""
    if len(probs) < 2:
        return probs[0] if probs else 0.0
    ordered = sorted(probs, reverse=True)
    return ordered[0] - ordered[1]


class ConformalRouter:
    """Choose the confidence threshold that hits a target error rate.

    Given a held-out calibration set of ``(confidence, correct)`` pairs, this
    finds a single threshold whose selected subset has selective risk at or below
    ``target_risk`` while answering as much traffic as possible. It is the
    conformal-risk-control view of a confidence threshold: you state the error
    rate you are willing to ship, and the threshold follows.

    Caveats worth knowing before trusting this in production:

    * It assumes the calibration examples are exchangeable with production
      traffic. Distribution shift voids the guarantee, not just the estimate.
    * Finite-sample noise means the realized risk on new data fluctuates around
      the target; keep the calibration set reasonably large.
    * If no threshold reaches the target, ``reachable`` is False and the router
      abstains on everything rather than silently over-promising.
    """

    def __init__(self, target_risk: float = 0.05) -> None:
        if not 0.0 < target_risk < 1.0:
            raise ValueError("target_risk must be in (0, 1)")
        self.target_risk = float(target_risk)
        self.threshold_: float | None = None
        self.reachable: bool = False
        self.calibration_coverage_: float = 0.0
        self.calibration_risk_: float = float("nan")
        self.fitted_: bool = False

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> "ConformalRouter":
        """Fit on held-out data. Higher confidence must mean more likely correct."""
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must be the same length")
        if not confidences:
            raise ValueError("need at least one calibration example")
        if not any(correct):
            raise ValueError(
                "calibration set has no correct examples; a risk target is unreachable"
            )

        n = len(confidences)
        pairs = sorted(zip(confidences, correct), key=lambda pair: -pair[0])

        selected = 0
        errors = 0
        best_threshold: float | None = None
        best_coverage = 0.0
        best_risk = float("nan")

        index = 0
        while index < n:
            threshold = pairs[index][0]
            # Add every example sharing this confidence, so the accepted set
            # exactly matches "confidence >= threshold".
            while index < n and pairs[index][0] == threshold:
                selected += 1
                if not pairs[index][1]:
                    errors += 1
                index += 1
            risk = errors / selected
            if risk <= self.target_risk:
                best_threshold = threshold
                best_coverage = selected / n
                best_risk = risk

        self.fitted_ = True
        if best_threshold is None:
            # Nothing reached the target: abstain on everything.
            self.threshold_ = math.inf
            self.reachable = False
            self.calibration_coverage_ = 0.0
            self.calibration_risk_ = float("nan")
        else:
            self.threshold_ = best_threshold
            self.reachable = True
            self.calibration_coverage_ = best_coverage
            self.calibration_risk_ = best_risk
        return self

    def route(self, confidences: Sequence[float]) -> list[bool]:
        """Return True (accept) / False (abstain) for each confidence."""
        self._require_fitted()
        return [c >= self.threshold_ for c in confidences]

    def evaluate(self, confidences: Sequence[float], correct: Sequence[bool]) -> tuple[float, float]:
        """Realized ``(risk, coverage)`` on any labeled set at the fitted threshold."""
        self._require_fitted()
        return selective_risk(confidences, correct, self.threshold_)

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("call fit() first")

    def __repr__(self) -> str:
        if not self.fitted_:
            return f"ConformalRouter(target_risk={self.target_risk:.3f}, unfitted)"
        return (
            f"ConformalRouter(target_risk={self.target_risk:.3f}, "
            f"threshold={self.threshold_:.4f}, "
            f"calibration_coverage={self.calibration_coverage_:.3f}, "
            f"calibration_risk={self.calibration_risk_:.4f})"
        )
