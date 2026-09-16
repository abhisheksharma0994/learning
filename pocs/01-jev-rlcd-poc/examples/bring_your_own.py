#!/usr/bin/env python3
"""Template: point this at your own decisions and read off the report.

Three sections to edit, marked below. Everything after them is the pipeline and
does not need to change.

    python examples/bring_your_own.py

Out of the box it runs on the built-in simulated model, so you can see the shape
of the output before wiring in real data.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.calibrate import HistogramBinner, TemperatureScaler  # noqa: E402
from jev_rlcd_poc.decide import ConformalRouter  # noqa: E402
from jev_rlcd_poc.metrics import argmax, ece, mce, nll, top1_accuracy  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1. YOUR LABELS. Every possible outcome, declared up front. Keep them short:
#    with the local backend each one must be a single token.
# ─────────────────────────────────────────────────────────────────────────────
LABELS = LabelSet(("billing", "shipping", "returns", "account", "other"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. YOUR DATA. You need labeled decisions -- cases where you already know the
#    right answer. That is the price of calibrated confidence, and there is no
#    way around it: without known answers you cannot measure, let alone fix,
#    whether the stated confidence is true.
# ─────────────────────────────────────────────────────────────────────────────
# Stub state, used only by the two default bodies below. Delete both when you
# wire in real data and a real scorer.
_STUB: dict = {}


def load_data(split: str) -> tuple[list[str], list[int]]:
    """Return ``(states, true_label_indices)`` for split ``"cal"`` or ``"test"``.

    Replace the body with your own loader. The two splits must not overlap.
    """
    # The stub must agree with LABELS above, or the check in main() will (rightly)
    # complain -- which is worth seeing once, so try changing LABELS and re-running.
    task = _STUB.setdefault("task", SyntheticDecisionTask(seed=0, n_labels=len(LABELS)))
    return task.sample(4000, seed=1 if split == "cal" else 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. YOUR SCORER. Anything that returns a probability per label, in label-set
#    order, for one state. Two ways to do it:
#
#    a) Local model, one forward pass for the whole label set:
#
#       from jev_rlcd_poc.backends.hf_local import LocalLabelScorer
#       scorer = LocalLabelScorer(LABELS, model_name="Qwen/Qwen2.5-0.5B-Instruct")
#
#    b) Your own callable -- an API, a classifier, a heuristic:
#
#       from jev_rlcd_poc.scorer import FunctionScorer
#       scorer = FunctionScorer(LABELS, my_function)   # -> {"billing": 0.7, ...}
#                                                      # or [0.7, 0.1, 0.1, 0.05, 0.05]
#
#       def my_function(state):
#           probs = my_api_classify(state)      # your code
#           return probs
# ─────────────────────────────────────────────────────────────────────────────
def build_scorer():
    """The default stub: the simulated model behind ``load_data``."""
    return _STUB["task"].scorer()


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-risk", type=float, default=0.05, help="error rate you are willing to ship")
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    cal_states, cal_y = load_data("cal")
    test_states, test_y = load_data("test")
    scorer = build_scorer()

    # Sanity checks first: bad labels or a mismatched scorer produce nonsense
    # metrics rather than an error, so catch them here.
    for split, targets in (("cal", cal_y), ("test", test_y)):
        bad = {target for target in targets if not 0 <= target < len(LABELS)}
        if bad:
            raise ValueError(
                f"{split} contains label indices {sorted(bad)} but the label set has "
                f"{len(LABELS)} entries; map your labels with LABELS.index_of(name)"
            )

    raw_cal, raw_test = scorer.score(cal_states), scorer.score(test_states)

    scaler = TemperatureScaler().fit(raw_cal, cal_y)
    binner = HistogramBinner(n_bins=args.bins).fit(raw_cal, cal_y)

    def report(name, probs):
        confidences = [max(row) for row in probs]
        return [
            name,
            f"{top1_accuracy(probs, test_y):.3f}",
            f"{ece(probs, test_y, n_bins=args.bins):.4f}",
            f"{mce(probs, test_y, n_bins=args.bins):.4f}",
            f"{nll(probs, test_y):.4f}",
            f"{sum(confidences) / len(confidences):.3f}",
        ]

    print("=" * 78)
    print("1. IS THE CONFIDENCE REAL?")
    print("=" * 78)
    print(f"{len(cal_states)} calibration decisions, {len(test_states)} held-out decisions")
    print(f"measured accuracy on held-out data: {top1_accuracy(raw_test, test_y):.3f}")
    print()
    print_table(
        ["variant", "accuracy", "ece", "mce", "nll", "mean confidence"],
        [
            report("raw", raw_test),
            report(f"temperature (T={scaler.temperature:.2f})", scaler.transform(raw_test)),
            report(f"histogram ({len(binner.edges_)} bins)", binner.transform(raw_test)),
        ],
    )
    print()
    print("Prefer the temperature row unless its ECE is clearly worse; the histogram")
    print("row fits one accuracy per bin and needs more calibration data to be stable.")
    print()

    best_raw = ece(raw_test, test_y, n_bins=args.bins)
    temp_test = scaler.transform(raw_test)
    bin_test = binner.transform(raw_test)
    best = min(
        [(ece(temp_test, test_y, n_bins=args.bins), "temperature", temp_test),
         (ece(bin_test, test_y, n_bins=args.bins), "histogram", bin_test)]
    )
    print(f"Raw ECE {best_raw:.4f} -> calibrated ECE {best[0]:.4f} ({best[1]})")
    if best_raw < 0.05:
        print("The raw confidence was already usable; calibrating bought little.")
    elif best[0] > 0.05:
        print("WARNING: still miscalibrated after fitting. More calibration data, or a")
        print("different model, is the next thing to try -- not a better threshold.")
    print()

    calibrated_test = best[2]
    calibrated_cal = (
        scaler.transform(raw_cal) if best[1] == "temperature" else binner.transform(raw_cal)
    )

    print("=" * 78)
    print(f"2. THE SHIPPING RULE YOU ASKED FOR (target {args.target_risk:.0%} error)")
    print("=" * 78)
    rows = []
    for target in (0.01, 0.02, 0.05, 0.10, 0.20):
        router = ConformalRouter(target_risk=target).fit(
            [max(row) for row in calibrated_cal],
            [argmax(row) == y for row, y in zip(calibrated_cal, cal_y)],
        )
        risk, coverage = router.evaluate(
            [max(row) for row in calibrated_test],
            [argmax(row) == y for row, y in zip(calibrated_test, test_y)],
        )
        rows.append(
            [
                f"{target:.0%}",
                "unreachable" if not router.reachable else f"{router.threshold_:.3f}",
                f"{router.calibration_coverage_:.1%}",
                f"{coverage:.1%}",
                "-" if math.isnan(risk) else f"{risk:.3f}",
            ]
        )
    print_table(
        ["target", "accept if >= ", "coverage (fit)", "coverage (held out)", "risk (held out)"],
        rows,
    )
    print()
    print("The first coverage column is what the calibration split promised; the second")
    print("is what actually happened on data the threshold never saw. A large gap means")
    print("your two splits are not exchangeable, and the guarantee does not transfer.")
    print()

    fitted = ConformalRouter(target_risk=args.target_risk).fit(
        [max(row) for row in calibrated_cal],
        [argmax(row) == y for row, y in zip(calibrated_cal, cal_y)],
    )
    if not fitted.reachable:
        print(f"No threshold reached {args.target_risk:.0%}. The model is never that reliable")
        print("at any confidence; escalate every decision to a human or a stronger model.")
    else:
        risk, coverage = fitted.evaluate(
            [max(row) for row in calibrated_test],
            [argmax(row) == y for row, y in zip(calibrated_test, test_y)],
        )
        print("Rule to ship:")
        print(f"    accept if confidence >= {fitted.threshold_:.3f}")
        print("    else escalate")
        print(f"  expected: {coverage:.1%} handled automatically at {risk:.1%} error")
    print()
    print("Re-run this after any model, prompt, or traffic change. A threshold is a")
    print("measurement, not a setting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
