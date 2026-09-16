"""Calibration and selective-prediction metrics.

Everything here is pure standard library so the evaluation harness runs anywhere.

Vocabulary: ``probs[i]`` is the model's full distribution over a fixed label set
for example ``i``, and ``y[i]`` is the index of the correct label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "ReliabilityBin",
    "adaptive_ece",
    "argmax",
    "aurc",
    "brier_score",
    "classwise_ece",
    "confidence",
    "ece",
    "mce",
    "nll",
    "reliability_bins",
    "risk_coverage_curve",
    "selective_risk",
    "top1_accuracy",
]


def _check(probs: Sequence[Sequence[float]], y: Sequence[int]) -> None:
    if len(probs) != len(y):
        raise ValueError(f"probs has {len(probs)} rows but y has {len(y)} labels")
    if not probs:
        raise ValueError("need at least one example")
    n_classes = len(probs[0])
    if n_classes == 0:
        raise ValueError("each probability row must have at least one entry")
    for i, row in enumerate(probs):
        if len(row) != n_classes:
            raise ValueError(f"row {i} has {len(row)} entries, expected {n_classes}")
        if any(p < 0.0 for p in row):
            raise ValueError(f"row {i} contains a negative probability")


def argmax(row: Sequence[float]) -> int:
    """Index of the largest entry. Ties go to the lowest index."""
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def confidence(row: Sequence[float]) -> float:
    """Top probability, i.e. the model's stated confidence."""
    return max(row)


def top1_accuracy(probs: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    """Fraction of examples where the argmax label is correct."""
    _check(probs, y)
    hits = sum(1 for row, target in zip(probs, y) if argmax(row) == target)
    return hits / len(probs)


def nll(probs: Sequence[Sequence[float]], y: Sequence[int], eps: float = 1e-12) -> float:
    """Mean negative log likelihood of the correct label."""
    _check(probs, y)
    total = 0.0
    for row, target in zip(probs, y):
        total -= math.log(max(row[target], eps))
    return total / len(probs)


def brier_score(probs: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    """Mean multiclass Brier score (squared error against the one-hot label)."""
    _check(probs, y)
    total = 0.0
    for row, target in zip(probs, y):
        for k, p in enumerate(row):
            total += (p - (1.0 if k == target else 0.0)) ** 2
    return total / len(probs)


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Confidence minus accuracy within this bin.

        Positive means overconfident: the model claimed more than it delivered.
        """
        return self.mean_confidence - self.accuracy


def reliability_bins(
    probs: Sequence[Sequence[float]],
    y: Sequence[int],
    n_bins: int = 15,
    adaptive: bool = False,
) -> list[ReliabilityBin]:
    """Bin examples by confidence and compare confidence against accuracy.

    ``adaptive=True`` uses equal-mass bins, which is the fairer view when
    confidences cluster near 1.0 -- the usual failure mode for LLMs. Ties in
    confidence are kept together, so a bin never mixes two halves of the same
    stated confidence with different accuracies.
    """
    _check(probs, y)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    pairs = [
        (confidence(row), argmax(row) == target)
        for row, target in zip(probs, y)
    ]

    if adaptive:
        # Equal-mass bins, but a run of identical confidences is never split
        # across two bins. Splitting a tie would compare the same stated
        # confidence against different realized accuracies, which reports
        # miscalibration where there is only an arbitrary ordering inside a tie.
        ordered = sorted(pairs, key=lambda pair: pair[0])
        tie_groups: list[tuple[float, float, list[tuple[float, bool]]]] = []
        i = 0
        while i < len(ordered):
            j = i
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            tie_groups.append((ordered[i][0], ordered[j - 1][0], ordered[i:j]))
            i = j

        target_mass = len(ordered) / n_bins
        groups = []
        pending: list[tuple[float, bool]] = []
        pending_lo = tie_groups[0][0] if tie_groups else 0.0
        for lo, hi, members in tie_groups:
            if pending and len(pending) + len(members) > target_mass:
                groups.append((pending_lo, pending[-1][0], pending))
                pending = []
                pending_lo = lo
            pending.extend(members)
        if pending:
            groups.append((pending_lo, pending[-1][0], pending))
    else:
        groups = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            # Half-open [lo, hi) so each example lands in exactly one bin; the
            # top bin is closed so confidence == 1.0 is included.
            members = (
                [pair for pair in pairs if lo <= pair[0] <= hi]
                if i == n_bins - 1
                else [pair for pair in pairs if lo <= pair[0] < hi]
            )
            groups.append((lo, hi, members))

    bins: list[ReliabilityBin] = []
    for lo, hi, members in groups:
        if not members:
            continue
        count = len(members)
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=count,
                mean_confidence=sum(c for c, _ in members) / count,
                accuracy=sum(1 for _, ok in members if ok) / count,
            )
        )
    return bins


def ece(
    probs: Sequence[Sequence[float]],
    y: Sequence[int],
    n_bins: int = 15,
    adaptive: bool = False,
) -> float:
    """Expected Calibration Error: |confidence - accuracy| averaged over bins.

    0.0 is perfect. A model that answers at 0.9 confidence but is right 60% of
    the time has an ECE around 0.30 and cannot be trusted to automate anything.
    """
    bins = reliability_bins(probs, y, n_bins=n_bins, adaptive=adaptive)
    total = sum(bin_.count for bin_ in bins)
    if total == 0:
        return 0.0
    return sum(bin_.count / total * abs(bin_.gap) for bin_ in bins)


def adaptive_ece(probs: Sequence[Sequence[float]], y: Sequence[int], n_bins: int = 15) -> float:
    """ECE with equal-mass bins. More stable when confidences cluster near 1.0."""
    return ece(probs, y, n_bins=n_bins, adaptive=True)


def mce(probs: Sequence[Sequence[float]], y: Sequence[int], n_bins: int = 15) -> float:
    """Maximum Calibration Error: the worst bin's gap."""
    bins = reliability_bins(probs, y, n_bins=n_bins)
    return max((abs(bin_.gap) for bin_ in bins), default=0.0)


def classwise_ece(
    probs: Sequence[Sequence[float]],
    y: Sequence[int],
    n_bins: int = 15,
    adaptive: bool = True,
) -> float:
    """ECE averaged over each label treated as a binary problem.

    Catches a model that is well calibrated on average but systematically wrong
    on one class -- common when the label set is high-cardinality.
    """
    _check(probs, y)
    n_classes = len(probs[0])
    scores = []
    for k in range(n_classes):
        if all(target != k for target in y):
            continue  # calibration is undefined for a class with no positives
        binary_probs = [[row[k], 1.0 - row[k]] for row in probs]
        binary_y = [0 if target == k else 1 for target in y]
        scores.append(ece(binary_probs, binary_y, n_bins=n_bins, adaptive=adaptive))
    return sum(scores) / len(scores) if scores else 0.0


def selective_risk(
    confidences: Sequence[float],
    correct: Sequence[bool],
    threshold: float,
) -> tuple[float, float]:
    """Risk and coverage when only confidence >= threshold decisions are shipped.

    Returns ``(risk, coverage)``; ``risk`` is NaN when nothing is answered. This
    is the operationally relevant pair: risk is the error rate you actually ship,
    coverage is how much traffic you handled automatically.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    selected = [ok for ok, c in zip(correct, confidences) if c >= threshold]
    if not selected:
        return float("nan"), 0.0
    return 1.0 - sum(selected) / len(selected), len(selected) / len(confidences)


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_points: int = 100,
) -> list[tuple[float, float]]:
    """Coverage/risk frontier from full coverage down to a single example.

    Requires an orderable confidence, which is exactly why calibration matters:
    a ranking only helps if confidence tracks correctness.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("need at least one example")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")

    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    curve: list[tuple[float, float]] = []
    errors = 0
    n = len(order)
    for rank, idx in enumerate(order, start=1):
        if not correct[idx]:
            errors += 1
        curve.append((rank / n, errors / rank))

    if n_points >= len(curve):
        return curve
    step = len(curve) / n_points
    picked = [curve[min(int(i * step), len(curve) - 1)] for i in range(n_points)]
    picked[-1] = curve[-1]
    return picked


def aurc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area Under the Risk-Coverage curve. Lower is better.

    A perfect ranker (every correct answer more confident than every wrong one)
    approaches 0; an uninformative ranker sits near the base error rate.
    """
    if not confidences:
        raise ValueError("need at least one example")
    curve = risk_coverage_curve(confidences, correct)
    area = 0.0
    prev_coverage, prev_risk = 0.0, curve[0][1]
    for coverage, risk in curve:
        area += (coverage - prev_coverage) * (prev_risk + risk) / 2.0
        prev_coverage, prev_risk = coverage, risk
    return area
