#!/usr/bin/env python3
"""What calibration actually buys you, measured end to end.

Runs with no GPU, no downloads and no API key: the simulated model in
``jev_rlcd_poc.backends.synthetic`` stands in for a real one. It is deliberately
overconfident in two different ways, so the report shows which calibrator fixes
which problem.

    python examples/demo_report.py
    python examples/demo_report.py --accuracy 0.6 --overconfidence 12 --print-bins
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.calibrate import HistogramBinner, TemperatureScaler  # noqa: E402
from jev_rlcd_poc.decide import ConformalRouter  # noqa: E402
from jev_rlcd_poc.metrics import (  # noqa: E402
    adaptive_ece,
    argmax,
    aurc,
    brier_score,
    ece,
    mce,
    nll,
    reliability_bins,
    top1_accuracy,
)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def summarize(name: str, probs, y, n_bins: int) -> list[str]:
    confidences = [max(row) for row in probs]
    correct = [argmax(row) == target for row, target in zip(probs, y)]
    return [
        name,
        f"{top1_accuracy(probs, y):.3f}",
        f"{ece(probs, y, n_bins=n_bins):.4f}",
        f"{adaptive_ece(probs, y, n_bins=n_bins):.4f}",
        f"{mce(probs, y, n_bins=n_bins):.4f}",
        f"{nll(probs, y):.4f}",
        f"{brier_score(probs, y):.4f}",
        f"{aurc(confidences, correct):.4f}",
    ]


def accuracy_above(probs, y, threshold: float):
    """Accuracy and share of traffic among decisions stated at >= threshold."""
    selected = [
        (max(row), argmax(row) == target)
        for row, target in zip(probs, y)
        if max(row) >= threshold
    ]
    if not selected:
        return None
    return sum(1 for _, ok in selected if ok) / len(selected), len(selected) / len(probs)


def confidences(probs) -> list[float]:
    return [max(row) for row in probs]


def correct_flags(probs, y) -> list[bool]:
    return [argmax(row) == target for row, target in zip(probs, y)]


def workflow_budget(probs, y, steps: int):
    """Predicted vs realized success for workflows of independent decisions.

    A workflow succeeds only if every step is right, so its true success
    probability is the product of the per-step correctness probabilities. If the
    per-step confidences are real probabilities, multiplying them predicts the
    total; if they are not, the prediction is fiction. This is the thing a
    threshold router cannot do for you, because a router yields accept/reject,
    not a probability you can multiply.
    """
    n_workflows = len(probs) // steps
    predicted = 0.0
    realized = 0
    for w in range(n_workflows):
        window = range(w * steps, (w + 1) * steps)
        product = 1.0
        for i in window:
            product *= max(probs[i])
        predicted += product
        realized += all(argmax(probs[i]) == y[i] for i in window)
    return predicted / n_workflows, realized / n_workflows


def bin_rows(probs, y, n_bins: int) -> list[list[str]]:
    return [
        [
            f"{bin_.lo:.2f}-{bin_.hi:.2f}",
            str(bin_.count),
            f"{bin_.mean_confidence:.3f}",
            f"{bin_.accuracy:.3f}",
            f"{bin_.gap:+.3f}",
        ]
        for bin_ in reliability_bins(probs, y, n_bins=n_bins)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8000, help="examples per split")
    parser.add_argument("--n-labels", type=int, default=32, help="size of the label set")
    parser.add_argument("--accuracy", type=float, default=0.72, help="how often the model is right")
    parser.add_argument("--overconfidence", type=float, default=8.0, help="power sharpening of stated confidence")
    parser.add_argument("--winner-bias", type=float, default=0.40, help="extra latent logit on the winner")
    parser.add_argument("--bins", type=int, default=15, help="bins for the reliability diagram")
    parser.add_argument("--print-bins", action="store_true", help="print the reliability diagram")
    args = parser.parse_args()

    task = SyntheticDecisionTask(
        n_labels=args.n_labels,
        accuracy=args.accuracy,
        overconfidence=args.overconfidence,
        winner_bias=args.winner_bias,
        seed=0,
    )
    cal_states, cal_y = task.sample(args.n, seed=1)
    test_states, test_y = task.sample(args.n, seed=2)
    scorer = task.scorer()
    raw_cal, raw_test = scorer.score(cal_states), scorer.score(test_states)

    # Calibrators are fitted on the calibration split and reported on the test
    # split, so nothing below is measured on data the calibrator has seen.
    scaler = TemperatureScaler().fit(raw_cal, cal_y)
    binner = HistogramBinner(n_bins=args.bins).fit(raw_cal, cal_y)
    temp_cal = scaler.transform(raw_cal)
    temp_test = scaler.transform(raw_test)
    bin_test = binner.transform(raw_test)
    variants = [
        ("raw", raw_test),
        (f"temperature (T={scaler.temperature:.2f})", temp_test),
        (f"histogram ({len(binner.edges_)} bins)", bin_test),
    ]

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    print(f"label set size            {args.n_labels}")
    print(f"calibration / test split  {args.n} / {args.n} decisions")
    print(f"true accuracy             {top1_accuracy(raw_test, test_y):.3f}  (asked for {args.accuracy:.2f})")
    print(f"mean stated confidence    {sum(confidences(raw_test)) / len(raw_test):.3f}")
    print()
    print("The model is right about three quarters of the time while claiming to be")
    print("certain. That gap is the whole reason calibrated confidence matters.")
    print()

    print("=" * 78)
    print("HELD-OUT PERFORMANCE BY VARIANT")
    print("=" * 78)
    print_table(
        ["variant", "acc", "ece", "adaptiveece", "mce", "nll", "brier", "aurc"],
        [summarize(name, probs, test_y, args.bins) for name, probs in variants],
    )
    print()
    print("Accuracy is unchanged by design -- both calibrators are monotone, so they")
    print("move confidence without moving decisions. ece/mce are the calibration")
    print("story; nll/brier are proper scoring rules that agree with it.")
    print()

    print("=" * 78)
    print("THE OPERATIONAL QUESTION")
    print("=" * 78)
    print("Of the decisions stated at >= 0.90 confidence, how many are actually right?")
    print()
    rows = []
    for name, probs in variants:
        result = accuracy_above(probs, test_y, 0.90)
        rows.append(
            [name, "none above 0.90", "-"]
            if result is None
            else [name, f"{result[0]:.3f}", f"{result[1]:.1%} of traffic"]
        )
    print_table(["variant", "accuracy at >=0.90", "coverage"], rows)
    print()
    print("Read the raw row as a warning: a 0.90 threshold looks like a 10% error")
    print("budget, and it is not one. Whichever way it lands, you cannot know in")
    print("advance -- you have to go and measure it.")
    print()

    print("=" * 78)
    print("WHAT CALIBRATION DOES **NOT** CHANGE")
    print("=" * 78)
    print("Temperature scaling is strictly monotone in confidence, so it preserves the")
    print("ranking of decisions exactly. Any method that only *orders* decisions -- a")
    print("confidence threshold, a risk-coverage curve, a router -- is therefore")
    print("invariant to it. Each column below fits its own router on its own data, and")
    print("they still land on the same operating points:")
    print()
    rows = []
    for target in (0.01, 0.05, 0.10, 0.20):
        row = [f"{target:.0%}"]
        for cal_probs, test_probs in ((raw_cal, raw_test), (temp_cal, temp_test)):
            router = ConformalRouter(target_risk=target).fit(
                confidences(cal_probs), correct_flags(cal_probs, cal_y)
            )
            risk, coverage = router.evaluate(
                confidences(test_probs), correct_flags(test_probs, test_y)
            )
            row += [f"{coverage:.1%}", "unreachable" if not router.reachable else f"{risk:.3f}"]
        rows.append(row)
    print_table(
        ["target risk", "cov raw", "risk raw", "cov cal", "risk cal"],
        rows,
    )
    print()
    print("The same operating points, reached from either column: a router gets")
    print("where you want to go without any calibration at all. The small gap is")
    print("granularity, not accuracy -- raw confidences are piled up against 1.0, so")
    print("the router has fewer distinct cut points to choose from in the tail. If")
    print("routing is all you need, you can stop reading here.")
    print()

    print("=" * 78)
    print("WHERE CALIBRATION ACTUALLY PAYS: COMPOSITION")
    print("=" * 78)
    print("Real workflows chain decisions. A workflow succeeds only if every step is")
    print("right, so its success probability is the product of the per-step ones.")
    print("Multiplying confidences is only sensible if they are probabilities.")
    print()
    rows = []
    for steps in (1, 5, 10, 20):
        row = [str(steps)]
        for name, probs in (variants[0], variants[2]):
            predicted, realized = workflow_budget(probs, test_y, steps)
            row += [f"{predicted:.3f}", f"{realized:.3f}"]
        rows.append(row)
    print_table(
        ["steps", "raw predicted", "raw realized", "cal predicted", "cal realized"],
        rows,
    )
    print()
    print("The raw column keeps promising that a 20-step workflow almost always works")
    print("while it did not succeed a single time in hundreds of runs. The calibrated")
    print("column is the one you can plan against, and the only one whose numbers can")
    print("be multiplied, averaged, or fed into expected-cost arithmetic at all.")
    print()

    if args.print_bins:
        for name, probs in (variants[0], variants[2]):
            print("=" * 78)
            print(f"RELIABILITY DIAGRAM ({name}) -- gap = stated confidence - realized accuracy")
            print("=" * 78)
            print_table(["bucket", "count", "confidence", "accuracy", "gap"], bin_rows(probs, test_y, args.bins))
            print()

    print("Next: examples/hf_single_pass.py runs the same interface against a real")
    print("local model, reading every label's logit out of a single forward pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
