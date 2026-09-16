"""Post-hoc calibration.

Raw probabilities from a language model are almost always *overconfident*: they
say 0.95 when they are right 0.6 of the time. That gap is what makes a model
unusable for automation even when its accuracy is fine. These calibrators close
it using a held-out calibration set, without touching the model.

Both calibrators preserve the argmax, so accuracy is unchanged while confidence
becomes honest.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from ._mathx import golden_section_min, log_softmax, safe_log, softmax
from .metrics import argmax

__all__ = ["HistogramBinner", "TemperatureScaler"]


def _validate(probs: Sequence[Sequence[float]], y: Sequence[int], name: str) -> None:
    if len(probs) != len(y):
        raise ValueError(f"{name}: probs has {len(probs)} rows but y has {len(y)} labels")
    if not probs:
        raise ValueError(f"{name}: need at least one example")
    n_classes = len(probs[0])
    for i, (row, target) in enumerate(zip(probs, y)):
        if len(row) != n_classes:
            raise ValueError(f"{name}: row {i} has {len(row)} entries, expected {n_classes}")
        if not 0 <= target < n_classes:
            raise ValueError(f"{name}: label {target} out of range for {n_classes} classes")


class TemperatureScaler:
    """Single-parameter calibration: ``softmax(logits / T)``.

    Fitted by minimizing negative log likelihood over ``T``. It is the cheapest
    calibration method that reliably works, and for LLM-style overconfidence it
    usually recovers most of the available improvement.

    Working from probabilities is equivalent to working from logits: for a
    softmax output, ``log(p)`` is the logit vector shifted by a constant per row,
    and softmax is invariant to constant shifts, so ``softmax(log(p) / T)``
    equals ``softmax(logits / T)``. Hence one class handles both representations.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.fitted_ = False
        self.calibration_nll_: float | None = None

    def fit_from_logits(
        self,
        logits: Sequence[Sequence[float]],
        y: Sequence[int],
        search_range: tuple[float, float] = (0.02, 50.0),
    ) -> "TemperatureScaler":
        """Fit T on raw (pre-softmax) logits."""
        if len(logits) != len(y):
            raise ValueError("logits and y must be the same length")
        if not logits:
            raise ValueError("need at least one example")
        if search_range[0] <= 0 or search_range[0] >= search_range[1]:
            raise ValueError("search_range must be a positive increasing pair")

        def nll_at(log_temperature: float) -> float:
            temperature = math.exp(log_temperature)
            total = 0.0
            for row, target in zip(logits, y):
                total -= log_softmax([value / temperature for value in row])[target]
            return total / len(logits)

        return self._search(nll_at, search_range)

    def fit(
        self,
        probs: Sequence[Sequence[float]],
        y: Sequence[int],
        search_range: tuple[float, float] = (0.02, 50.0),
    ) -> "TemperatureScaler":
        """Fit T on probabilities (equivalent to fitting on their logits)."""
        _validate(probs, y, "TemperatureScaler.fit")
        logits = [[safe_log(p) for p in row] for row in probs]
        return self.fit_from_logits(logits, y, search_range=search_range)

    def _search(
        self,
        nll_at: Callable[[float], float],
        search_range: tuple[float, float],
    ) -> "TemperatureScaler":
        # Coarse scan first: the NLL-vs-T curve is usually but not guaranteed to
        # be unimodal, so find the best basin and refine inside it.
        lo_log, hi_log = math.log(search_range[0]), math.log(search_range[1])
        n_grid = 81
        step = (hi_log - lo_log) / (n_grid - 1)
        grid = [(lo_log + i * step, nll_at(lo_log + i * step)) for i in range(n_grid)]
        best_idx = min(range(n_grid), key=lambda i: grid[i][1])
        left = grid[max(best_idx - 1, 0)][0]
        right = grid[min(best_idx + 1, n_grid - 1)][0]

        best_log_temperature, best_nll = golden_section_min(nll_at, left, right, iters=100)
        self.temperature = math.exp(best_log_temperature)
        self.calibration_nll_ = best_nll
        self.fitted_ = True
        return self

    def transform_from_logits(self, logits: Sequence[Sequence[float]]) -> list[list[float]]:
        self._require_fitted()
        return [softmax([value / self.temperature for value in row]) for row in logits]

    def transform(self, probs: Sequence[Sequence[float]]) -> list[list[float]]:
        """Rescale probabilities. Preserves the argmax."""
        self._require_fitted()
        logits = [[safe_log(p) for p in row] for row in probs]
        return self.transform_from_logits(logits)

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("call fit() or fit_from_logits() first")

    def __repr__(self) -> str:
        state = f"T={self.temperature:.4f}" if self.fitted_ else "unfitted"
        return f"TemperatureScaler({state})"


class HistogramBinner:
    """Non-parametric calibration: map confidence to empirical accuracy.

    Equal-mass bins of the calibration set each contribute their empirical
    accuracy, and the bin accuracies are forced non-decreasing so the mapping is
    monotone in the stated confidence. Because it learns one accuracy per bin
    rather than a single global temperature, it can absorb miscalibration that
    no temperature can -- at the cost of needing more calibration data, since
    thin bins are noisy.
    """

    def __init__(self, n_bins: int = 15) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.n_bins = n_bins
        self.edges_: list[float] = []
        self.accuracies_: list[float] = []
        self.counts_: list[int] = []
        self.fitted_ = False

    def fit(self, probs: Sequence[Sequence[float]], y: Sequence[int]) -> "HistogramBinner":
        _validate(probs, y, "HistogramBinner.fit")
        n = len(probs)
        order = sorted(range(n), key=lambda i: max(probs[i]))
        confidences = [max(probs[i]) for i in order]
        correct = [1 if argmax(probs[i]) == y[i] else 0 for i in order]

        edges: list[float] = []
        accuracies: list[float] = []
        counts: list[int] = []
        for b in range(self.n_bins):
            start, stop = (b * n) // self.n_bins, ((b + 1) * n) // self.n_bins
            if start >= stop:
                continue
            window = correct[start:stop]
            edges.append(confidences[start])
            accuracies.append(sum(window) / len(window))
            counts.append(len(window))

        for i in range(1, len(accuracies)):
            if accuracies[i] < accuracies[i - 1]:
                accuracies[i] = accuracies[i - 1]

        self.edges_ = edges
        self.accuracies_ = accuracies
        self.counts_ = counts
        self.fitted_ = True
        return self

    def _lookup(self, confidence: float) -> float:
        idx = 0
        for i, edge in enumerate(self.edges_):
            if confidence >= edge:
                idx = i
            else:
                break
        return self.accuracies_[idx]

    #: Bisection steps used to hit the target confidence for one row. 30 steps
    #: resolve the exponent to well under a millionth, which is far finer than
    #: the bin boundaries themselves.
    _RESCALE_STEPS = 30

    def _rescale(self, row: Sequence[float], target: float) -> list[float]:
        """Reshape a row until its top probability equals ``target``.

        The reshape is a power transform, ``softmax(alpha * log p)`` with
        ``alpha >= 0``, which is monotone in ``alpha`` and therefore cannot
        reorder the labels: the argmax that came out of the model is the argmax
        that ships. ``alpha`` is found by bisection.

        A side effect worth noting: it is impossible to express less confidence
        than uniform while keeping the ranking, so rows whose target falls below
        1/k are left alone.
        """
        logs = [safe_log(p) for p in row]
        peak = max(logs)
        shifted = [value - peak for value in logs]  # all <= 0, top is 0.0

        def top_probability(alpha: float) -> float:
            return 1.0 / sum(math.exp(alpha * value) for value in shifted)

        if len(row) == 1 or target <= top_probability(0.0):
            return list(row)  # already at or below the floor

        lo, hi = 0.0, 1.0
        while top_probability(hi) < target and hi < 1024.0:
            hi *= 2.0

        for _ in range(self._RESCALE_STEPS):
            mid = (lo + hi) / 2.0
            if top_probability(mid) < target:
                lo = mid
            else:
                hi = mid

        alpha = (lo + hi) / 2.0
        weights = [math.exp(alpha * value) for value in shifted]
        total = sum(weights)
        return [weight / total for weight in weights]

    def transform(self, probs: Sequence[Sequence[float]]) -> list[list[float]]:
        """Move each row's top probability to its bin's empirical accuracy."""
        self._require_fitted()
        return [self._rescale(row, self._lookup(max(row))) for row in probs]

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("call fit() first")

    def __repr__(self) -> str:
        state = f"{len(self.edges_)} bins" if self.fitted_ else "unfitted"
        return f"HistogramBinner({state})"
