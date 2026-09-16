"""Tests for the high-level decider: python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.calibrated import CalibratedDecider, CalibrationReport  # noqa: E402
from jev_rlcd_poc.metrics import argmax  # noqa: E402
from jev_rlcd_poc.scorer import FunctionScorer, LabelSet  # noqa: E402


def dataset(n_cal: int = 3000, n_test: int = 3000):
    task = SyntheticDecisionTask(seed=3, accuracy=0.75, overconfidence=8.0)
    cal_states, cal_y = task.sample(n_cal, seed=1)
    test_states, test_y = task.sample(n_test, seed=2)
    scorer = task.scorer()
    return scorer, cal_states, cal_y, test_states, test_y


class TestFit(unittest.TestCase):
    def test_reduces_ece_and_still_agrees_with_the_scorer(self):
        scorer, cal_states, cal_y, test_states, test_y = dataset()
        decider = CalibratedDecider.fit(scorer, cal_states, cal_y, target_risk=0.10)
        report = decider.evaluate(test_states, test_y)

        self.assertIsInstance(report, CalibrationReport)
        self.assertLess(report.calibrated_ece, report.raw_ece)
        self.assertGreater(report.raw_ece, 0.1)  # the synthetic model is overconfident

        # Calibration is monotone, so no decision changed.
        raw = scorer.score(test_states[:200])
        mismatches = sum(
            1
            for row, state in zip(raw, test_states[:200])
            if argmax(row) != argmax(decider.probabilities(state))
        )
        self.assertEqual(mismatches, 0)

    def test_histogram_method_also_calibrates(self):
        scorer, cal_states, cal_y, test_states, test_y = dataset()
        decider = CalibratedDecider.fit(
            scorer, cal_states, cal_y, target_risk=0.10, method="histogram", n_bins=10
        )
        report = decider.evaluate(test_states, test_y)
        self.assertLess(report.calibrated_ece, report.raw_ece)

    def test_reports_are_populated(self):
        scorer, cal_states, cal_y, test_states, test_y = dataset()
        report = CalibratedDecider.fit(scorer, cal_states, cal_y).evaluate(test_states, test_y)
        self.assertEqual(report.n_cal, len(cal_states))
        self.assertEqual(report.n_test, len(test_states))
        self.assertLessEqual(report.coverage, 1.0)
        self.assertGreater(report.accuracy, 0.5)
        text = str(report)
        self.assertIn("ECE raw -> calibrated", text)

    def test_decide_matches_the_router_threshold(self):
        scorer, cal_states, cal_y, test_states, test_y = dataset()
        decider = CalibratedDecider.fit(scorer, cal_states, cal_y, target_risk=0.05)
        for state in test_states[:60]:
            decision, probs, accepted = decider.decide(state)
            self.assertAlmostEqual(decision.confidence, max(probs), places=12)
            self.assertEqual(decision.label, decider.label_set[argmax(probs)])
            expected = decider.router.route([max(probs)])[0]
            self.assertEqual(accepted, expected)

    def test_looser_target_accepts_at_least_as_much(self):
        scorer, cal_states, cal_y, test_states, _ = dataset()
        strict = CalibratedDecider.fit(scorer, cal_states, cal_y, target_risk=0.01)
        loose = CalibratedDecider.fit(scorer, cal_states, cal_y, target_risk=0.30)
        # A stricter error target demands more confidence, so a higher threshold.
        self.assertGreaterEqual(strict.router.threshold_, loose.router.threshold_)
        accepted_strict = sum(strict.decide(s)[2] for s in test_states[:200])
        accepted_loose = sum(loose.decide(s)[2] for s in test_states[:200])
        self.assertLessEqual(accepted_strict, accepted_loose)


class TestValidation(unittest.TestCase):
    labels = LabelSet(("a", "b"))

    def scorer(self):
        return FunctionScorer(self.labels, lambda state: [0.8, 0.2])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            CalibratedDecider.fit(self.scorer(), ["s1", "s2"], [0])

    def test_empty_calibration_set(self):
        with self.assertRaises(ValueError):
            CalibratedDecider.fit(self.scorer(), [], [])

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            CalibratedDecider.fit(self.scorer(), ["s"], [0], method="magic")

    def test_scorer_without_a_label_set(self):
        class Bare:
            def score(self, states):
                return [[0.5, 0.5] for _ in states]

        with self.assertRaises(ValueError):
            CalibratedDecider.fit(Bare(), ["s"], [0])

    def test_evaluate_length_mismatch(self):
        decider = CalibratedDecider.fit(self.scorer(), ["s1", "s2"], [0, 0])
        with self.assertRaises(ValueError):
            decider.evaluate(["s1"], [0, 1])

    def test_repr_is_informative(self):
        decider = CalibratedDecider.fit(self.scorer(), ["s1", "s2"], [0, 1])
        self.assertIn("CalibratedDecider", repr(decider))


if __name__ == "__main__":
    unittest.main(verbosity=2)
