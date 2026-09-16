"""The decision interface: a fixed label set in, a probability per label out.

The contract that matters for automation is that the output space is *declared
in advance*. A string generator can emit anything and then has to be parsed and
validated; a scorer over a closed label set cannot produce a value outside the
set, so the type errors that plague agent tool-calls cannot occur at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ._mathx import softmax
from .metrics import argmax, confidence

__all__ = [
    "FunctionScorer",
    "LabelSet",
    "ScoredBatch",
    "Scorer",
    "chunks",
    "validate_prob_rows",
]


@dataclass(frozen=True)
class LabelSet:
    """A closed, ordered set of decision outcomes.

    Order is meaningful: probability row index ``i`` refers to ``labels[i]``.
    Keep labels to a single token if you want truly parallel single-pass scoring
    (see ``backends.hf_local``).
    """

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("a LabelSet needs at least one label")
        if len(set(self.labels)) != len(self.labels):
            duplicates = sorted({label for label in self.labels if self.labels.count(label) > 1})
            raise ValueError(f"duplicate labels: {duplicates}")

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __getitem__(self, index: int) -> str:
        return self.labels[index]

    def index_of(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(f"{label!r} is not in the label set") from None

    @classmethod
    def from_iterable(cls, labels: Iterable[str]) -> "LabelSet":
        return cls(tuple(labels))


@runtime_checkable
class Scorer(Protocol):
    """Anything that can turn a structured state into a distribution over labels.

    Implementations must return rows of length ``len(label_set)`` that sum to 1,
    in label-set order, for every input state.
    """

    @property
    def label_set(self) -> LabelSet: ...

    def score(self, states: Sequence[str]) -> list[list[float]]: ...


class FunctionScorer:
    """Wrap any callable as a :class:`Scorer`.

    This is the seam for using the harness with something that is not the local
    backend: a hosted API, an existing classifier, a heuristic, or a test double.
    The callable takes one state and returns either a mapping of label to score,
    or a sequence in label-set order.

    ``mode`` declares what the callable returns:

    * ``"probability"`` (default) -- probabilities. They are renormalized by
      default, since hand-written scores rarely sum to exactly one; pass
      ``normalize=False`` to have anything that is not a proper distribution
      rejected instead.
    * ``"logits"`` -- unnormalized scores, so negatives are expected. They are
      exponentiated with a max shift and then normalized, which is how hosted
      logprobs and raw model logits are turned into a distribution.
    """

    def __init__(
        self,
        label_set: LabelSet,
        function: Callable[[str], Mapping[str, float] | Sequence[float]],
        *,
        mode: str = "probability",
        normalize: bool = True,
        tolerance: float = 1e-6,
    ) -> None:
        if mode not in ("probability", "logits"):
            raise ValueError("mode must be 'probability' or 'logits'")
        self.label_set = label_set
        self.function = function
        self.mode = mode
        self.normalize = normalize
        self.tolerance = tolerance

    def _to_scores(self, raw: Mapping[str, float] | Sequence[float], state: str) -> list[float]:
        if isinstance(raw, Mapping):
            missing = [label for label in self.label_set if label not in raw]
            if missing:
                raise ValueError(f"scorer for {state!r} did not return labels {missing}")
            return [float(raw[label]) for label in self.label_set]
        scores = [float(value) for value in raw]
        if len(scores) != len(self.label_set):
            raise ValueError(
                f"scorer for {state!r} returned {len(scores)} values, "
                f"expected {len(self.label_set)}"
            )
        return scores

    def _to_row(self, raw: Mapping[str, float] | Sequence[float], state: str) -> list[float]:
        scores = self._to_scores(raw, state)

        if self.mode == "logits":
            if not any(value > float("-inf") for value in scores):
                raise ValueError(f"scorer for {state!r} returned no usable logits")
            return softmax(scores)

        if any(value < 0.0 for value in scores):
            raise ValueError(
                f"scorer for {state!r} returned a negative probability; "
                "pass mode='logits' if these are logprobs"
            )
        total = sum(scores)
        if total <= 0.0:
            raise ValueError(f"scorer for {state!r} returned no probability mass")
        if self.normalize:
            return [value / total for value in scores]
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                f"scorer for {state!r} returned probabilities summing to {total:.6f}; "
                "pass normalize=True to renormalize them"
            )
        return scores

    def score(self, states: Sequence[str]) -> list[list[float]]:
        return [self._to_row(self.function(state), state) for state in states]

    def __repr__(self) -> str:
        return (
            f"FunctionScorer(labels={len(self.label_set)}, mode={self.mode!r}, "
            f"normalize={self.normalize})"
        )


def validate_prob_rows(rows: Sequence[Sequence[float]], label_set: LabelSet, tol: float = 1e-6) -> None:
    """Raise if a scorer returned the wrong shape or unnormalized rows."""
    for i, row in enumerate(rows):
        if len(row) != len(label_set):
            raise ValueError(
                f"row {i} has {len(row)} probabilities but the label set has {len(label_set)} labels"
            )
        if any(p < 0.0 or p > 1.0 for p in row):
            raise ValueError(f"row {i} has a probability outside [0, 1]")
        total = sum(row)
        if abs(total - 1.0) > tol:
            raise ValueError(f"row {i} sums to {total:.6f}, expected 1.0")


@dataclass
class ScoredBatch:
    """Probabilities from one scoring pass, plus what it cost.

    ``latency_ms`` is wall-clock for the batch. For a true single-pass scorer the
    per-decision cost is roughly ``latency_ms / len(states)``, because every
    label's probability came out of the same forward pass.
    """

    states: list[str]
    probs: list[list[float]]
    latency_ms: float
    label_set: LabelSet
    forward_passes: int = 0
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def ms_per_decision(self) -> float:
        return self.latency_ms / len(self.states) if self.states else 0.0

    def confidences(self) -> list[float]:
        return [confidence(row) for row in self.probs]

    def label_indices(self) -> list[int]:
        return [argmax(row) for row in self.probs]

    def labels(self) -> list[str]:
        return [self.label_set[i] for i in self.label_indices()]

    def correct(self, y: Sequence[int]) -> list[bool]:
        """Per-example correctness, for the calibration and risk-coverage views."""
        if len(y) != len(self.probs):
            raise ValueError("y must be the same length as the batch")
        return [argmax(row) == target for row, target in zip(self.probs, y)]


def chunks(items: Sequence, size: int):
    """Yield consecutive slices of at most ``size`` items."""
    if size < 1:
        raise ValueError("size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]
