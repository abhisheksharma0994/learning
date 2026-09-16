"""Minimal numeric helpers. Standard library only, on purpose."""

from __future__ import annotations

import math
from typing import Callable, Sequence

__all__ = ["golden_section_min", "log_softmax", "logsumexp", "softmax"]

_LOG_EPS = 1e-12


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    if not values:
        raise ValueError("logsumexp of an empty sequence")
    peak = max(values)
    if peak == float("-inf"):
        return float("-inf")
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def softmax(values: Sequence[float]) -> list[float]:
    """Softmax over a sequence of logits."""
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(values: Sequence[float]) -> list[float]:
    """Log of softmax, computed stably."""
    lse = logsumexp(values)
    return [v - lse for v in values]


def golden_section_min(
    fn: Callable[[float], float],
    lo: float,
    hi: float,
    iters: int = 80,
    tol: float = 1e-9,
) -> tuple[float, float]:
    """Minimize a unimodal function on [lo, hi]. Returns ``(x, fn(x))``.

    Used to fit a single temperature, which keeps calibration dependency-free.
    """
    if lo >= hi:
        raise ValueError("require lo < hi")
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # 1/phi ~ 0.618
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = fn(d)
    best = (a + b) / 2.0
    return best, fn(best)


def safe_log(p: float) -> float:
    """log with a floor, so a zero probability does not produce -inf."""
    return math.log(max(p, _LOG_EPS))
