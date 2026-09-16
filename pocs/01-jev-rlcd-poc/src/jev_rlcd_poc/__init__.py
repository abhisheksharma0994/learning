"""Jev RLCD POC: a harness for typed, calibrated decisions.

The idea (borrowed from TypeSafe's Jev / System One pitch): instead of
generating strings and parsing them, constrain a model to emit a choice over a
fixed, pre-declared label set, attach a probability to every option, and then
verify that those probabilities are actually *calibrated* -- i.e. that when the
model says 0.9 it is right about 90% of the time.

This package does three separable things:

1. ``scorer`` / ``backends``  -- get a full probability distribution over the
   label set in a single forward pass (no autoregressive string generation).
2. ``calibrate``             -- make those probabilities honest after the fact.
3. ``metrics`` / ``decide``  -- prove it, and route low-confidence decisions to
   a human or a bigger model instead of guessing.

Only the standard library is required. Install the ``local`` extra for the
HuggingFace single-pass backend.
"""

from .calibrate import HistogramBinner, TemperatureScaler
from .calibrated import CalibratedDecider, CalibrationReport
from .decide import ConformalRouter, Decision, select
from .metrics import (
    adaptive_ece,
    aurc,
    brier_score,
    classwise_ece,
    ece,
    nll,
    reliability_bins,
    risk_coverage_curve,
)
from .scorer import FunctionScorer, LabelSet, ScoredBatch, Scorer

__all__ = [
    "CalibratedDecider",
    "CalibrationReport",
    "ConformalRouter",
    "Decision",
    "FunctionScorer",
    "HistogramBinner",
    "LabelSet",
    "ScoredBatch",
    "Scorer",
    "TemperatureScaler",
    "adaptive_ece",
    "aurc",
    "brier_score",
    "classwise_ece",
    "ece",
    "nll",
    "reliability_bins",
    "risk_coverage_curve",
    "select",
]
