"""Tests for the metrics. Run with: python -m unittest discover -s tests"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.metrics import (  # noqa: E402
    adaptive_ece,
    argmax,
    aurc,
    brier_score,
    classwise_ece,
    confidence,
    ece,
    mce,
    nll,
    reliability_bins,
    risk_coverage_curve,
    selective_risk,
    top1_accuracy,
)


def calibrated_dataset(accuracy_at_high: float, accuracy_at_low: float):
    """Build a dataset whose confidence is exactly its accuracy per group."""
    n_high, n_low = 80, 20
    probs, y = [], []
    for i in range(n_high):
        correct = i < round(n_high * accuracy_at_high)
        probs.append([0.8, 0.2])
        y.append(0 if correct else 1)
    for i in range(n_low):
        correct = i < round(n_low * accuracy_at_low)
        probs.append([0.6, 0.4])
        y.append(0 if correct else 1)
    return probs, y


class TestBasicMetrics(unittest.TestCase):
    def test_argmax_and_confidence(self):
        self.assertEqual(argmax([0.1, 0.7, 0.2]), 1)
        self.assertEqual(argmax([0.5, 0.5]), 0)  # ties go to the lowest index
        self.assertAlmostEqual(confidence([0.1, 0.7, 0.2]), 0.7)

    def test_top1_accuracy(self):
        probs = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
        y = [0, 1, 1]
        self.assertAlmostEqual(top1_accuracy(probs, y), 2 / 3)

    def test_nll_and_brier_on_uniform(self):
        probs = [[0.5, 0.5]]
        y = [0]
        self.assertAlmostEqual(nll(probs, y), math.log(2))
        self.assertAlmostEqual(brier_score(probs, y), 0.5)

    def test_brier_is_zero_for_a_confident_correct_answer(self):
        self.assertAlmostEqual(brier_score([[1.0, 0.0]], [0]), 0.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            ece([[0.5, 0.5]], [0, 1])
        with self.assertRaises(ValueError):
            ece([[0.5, 0.5, 0.0], [0.5, 0.3]], [0, 1])  # rows of different length
        with self.assertRaises(ValueError):
            ece([[-0.1, 1.1]], [0])
        with self.assertRaises(ValueError):
            ece([], [])


class TestCalibrationMetrics(unittest.TestCase):
    def test_perfectly_calibrated_data_has_near_zero_ece(self):
        probs, y = calibrated_dataset(0.8, 0.6)
        self.assertLess(ece(probs, y, n_bins=10), 1e-9)
        self.assertLess(adaptive_ece(probs, y, n_bins=10), 1e-9)
        self.assertLess(mce(probs, y, n_bins=10), 1e-9)

    def test_overconfident_data_has_large_ece(self):
        probs, y = calibrated_dataset(0.5, 0.3)
        # High-confidence group states 0.8 and delivers 0.5 -> gap 0.3 over 80%.
        self.assertAlmostEqual(ece(probs, y, n_bins=10), 0.8 * 0.3 + 0.2 * 0.3, places=6)
        self.assertAlmostEqual(mce(probs, y, n_bins=10), 0.3, places=6)

    def test_reliability_bins_cover_every_example_once(self):
        probs, y = calibrated_dataset(0.8, 0.6)
        bins = reliability_bins(probs, y, n_bins=5)
        self.assertEqual(sum(b.count for b in bins), len(probs))
        self.assertTrue(all(b.lo < b.hi for b in bins))

    def test_adaptive_bins_are_near_balanced_when_confidences_are_distinct(self):
        probs = [[0.5 + i / 1000, 0.5 - i / 1000] for i in range(100)]
        y = [0] * 100
        bins = reliability_bins(probs, y, n_bins=4, adaptive=True)
        counts = [b.count for b in bins]
        self.assertEqual(sum(counts), len(probs))
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_adaptive_bins_never_split_a_confidence_tie(self):
        # 80 examples at 0.8 (the correct ones listed first) and 20 at 0.6. A
        # blind equal-mass split at 25 would cut the 0.8 group in half, compare
        # each half against 1.0 and 0.4 accuracy, and report a 0.32 calibration
        # error on data that has none.
        probs, y = [], []
        for i in range(80):
            probs.append([0.8, 0.2])
            y.append(0 if i < 64 else 1)
        for i in range(20):
            probs.append([0.6, 0.4])
            y.append(0 if i < 12 else 1)
        bins = reliability_bins(probs, y, n_bins=4, adaptive=True)
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(adaptive_ece(probs, y, n_bins=4), 0.0, places=9)

    def test_gap_sign_means_overconfidence(self):
        probs, y = calibrated_dataset(0.5, 0.5)
        for bin_ in reliability_bins(probs, y, n_bins=10):
            self.assertGreater(bin_.gap, 0.0)

    def test_classwise_ece_surfaces_per_class_miscalibration(self):
        # Top-1 confidence is nearly right (0.7 stated, 0.6 realized), but class 1
        # is handed 0.15 while actually showing up 40% of the time. The global
        # number understates how wrong that one coordinate is.
        probs = [[0.7, 0.15, 0.15]] * 100
        y = [0] * 60 + [1] * 40
        self.assertAlmostEqual(ece(probs, y), 0.1, places=9)
        self.assertGreater(classwise_ece(probs, y), 1.5 * ece(probs, y))

    def test_classwise_ece_skips_unobserved_classes(self):
        # Only class 0 occurs, so classes 1 and 2 must not contribute. If they
        # did, the average would be pulled down to ~0.133.
        self.assertAlmostEqual(classwise_ece([[0.8, 0.1, 0.1]] * 10, [0] * 10), 0.2)


class TestSelectivePrediction(unittest.TestCase):
    def test_selective_risk_and_coverage(self):
        confidences = [0.9, 0.8, 0.3]
        correct = [True, False, True]
        risk, coverage = selective_risk(confidences, correct, 0.8)
        self.assertAlmostEqual(risk, 0.5)  # one error out of two answered
        self.assertAlmostEqual(coverage, 2 / 3)

    def test_selective_risk_is_nan_when_nothing_is_answered(self):
        risk, coverage = selective_risk([0.4], [True], 0.9)
        self.assertTrue(math.isnan(risk))
        self.assertEqual(coverage, 0.0)

    def test_risk_coverage_curve_starts_at_and_ends_at_full_error_rate(self):
        confidences = [0.9, 0.5, 0.7, 0.2]
        correct = [True, False, True, False]
        curve = risk_coverage_curve(confidences, correct)
        self.assertAlmostEqual(curve[-1][0], 1.0)
        self.assertAlmostEqual(curve[-1][1], 0.5)
        coverages = [coverage for coverage, _ in curve]
        self.assertEqual(coverages, sorted(coverages))

    def test_aurc_rewards_a_better_ranking(self):
        correct = [True, True, False, False]
        good = [0.9, 0.8, 0.3, 0.2]   # errors ranked last
        bad = [0.2, 0.3, 0.8, 0.9]    # errors ranked first
        # A perfect ranking is not a zero-area curve: once every error has been
        # pulled in, the curve climbs from 0.5 coverage to full coverage.
        self.assertLess(aurc(good, correct), aurc(bad, correct) / 4)

    def test_aurc_validation(self):
        with self.assertRaises(ValueError):
            aurc([], [])
        with self.assertRaises(ValueError):
            aurc([0.5], [True, False])
        with self.assertRaises(ValueError):
            risk_coverage_curve([0.5], [True], n_points=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
