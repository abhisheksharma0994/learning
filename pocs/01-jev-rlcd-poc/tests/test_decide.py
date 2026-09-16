"""Tests for routing decisions: python -m unittest discover -s tests"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.calibrate import TemperatureScaler  # noqa: E402
from jev_rlcd_poc.decide import ConformalRouter, entropy, margin, normalized_entropy, select  # noqa: E402
from jev_rlcd_poc.metrics import argmax  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

LABELS = LabelSet(("page-00", "page-01", "page-02"))


def calibrated_split(target_risk: float = 0.05, n: int = 6000):
    """A calibrated dataset, split into a routing-fit half and an eval half."""
    task = SyntheticDecisionTask(seed=5, accuracy=0.72, overconfidence=2.5)
    cal_states, cal_y = task.sample(n, seed=1)
    test_states, test_y = task.sample(n, seed=2)
    scorer = task.scorer()
    scaler = TemperatureScaler().fit(scorer.score(cal_states), cal_y)

    cal_probs = scaler.transform(scorer.score(cal_states))
    test_probs = scaler.transform(scorer.score(test_states))
    cal_conf = [max(row) for row in cal_probs]
    test_conf = [max(row) for row in test_probs]
    cal_correct = [argmax(row) == y for row, y in zip(cal_probs, cal_y)]
    test_correct = [argmax(row) == y for row, y in zip(test_probs, test_y)]
    router = ConformalRouter(target_risk=target_risk).fit(cal_conf, cal_correct)
    return router, test_conf, test_correct


class TestSelect(unittest.TestCase):
    def test_picks_the_argmax_label_and_carries_confidence(self):
        decision = select([0.1, 0.7, 0.2], LABELS)
        self.assertEqual(decision.label, "page-01")
        self.assertEqual(decision.index, 1)
        self.assertAlmostEqual(decision.confidence, 0.7)
        self.assertEqual(decision.action, "accept")
        self.assertFalse(decision.abstained)

    def test_decision_is_immutable(self):
        decision = select([0.6, 0.3, 0.1], LABELS)
        with self.assertRaises(Exception):
            decision.label = "page-02"  # type: ignore[misc]


class TestProbabilityShapeHelpers(unittest.TestCase):
    def test_entropy_is_zero_for_a_certain_answer(self):
        self.assertAlmostEqual(entropy([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(entropy([0.5, 0.5]), math.log(2))

    def test_normalized_entropy_is_scale_free(self):
        self.assertAlmostEqual(normalized_entropy([0.5, 0.5]), 1.0)
        self.assertAlmostEqual(normalized_entropy([0.125] * 8), 1.0)
        self.assertAlmostEqual(normalized_entropy([1.0, 0.0]), 0.0)

    def test_margin_measures_ambiguity(self):
        self.assertAlmostEqual(margin([0.5, 0.3, 0.2]), 0.2)
        self.assertAlmostEqual(margin([0.4, 0.4]), 0.0)


class TestConformalRouter(unittest.TestCase):
    def test_hits_the_target_risk_on_held_out_data(self):
        router, test_conf, test_correct = calibrated_split(target_risk=0.05)
        risk, coverage = router.evaluate(test_conf, test_correct)
        self.assertTrue(router.reachable)
        self.assertGreater(coverage, 0.3)
        # Finite-sample slack: the guarantee is on the calibration set, not this
        # one, so allow the realized risk to drift a little above the target.
        self.assertLess(risk, 0.12)

    def test_higher_target_risk_buys_more_coverage(self):
        coverages = []
        for target in (0.02, 0.05, 0.10, 0.20):
            router, test_conf, test_correct = calibrated_split(target_risk=target)
            _, coverage = router.evaluate(test_conf, test_correct)
            coverages.append(coverage)
        self.assertEqual(coverages, sorted(coverages))
        self.assertEqual(len(set(coverages)), len(coverages))

    def test_accepts_almost_everything_when_the_target_is_trivially_loose(self):
        router, test_conf, test_correct = calibrated_split(target_risk=0.99)
        risk, coverage = router.evaluate(test_conf, test_correct)
        # The threshold is the lowest confidence seen during calibration, so a
        # handful of lower-confidence test cases are still abstained on.
        self.assertGreater(coverage, 0.98)
        self.assertGreater(risk, 0.0)

    def test_unreachable_target_abstains_on_everything(self):
        # One confidence value, and most of them wrong: no threshold can hit 1%.
        router = ConformalRouter(target_risk=0.01).fit([0.5] * 100, [True] * 20 + [False] * 80)
        self.assertFalse(router.reachable)
        self.assertEqual(router.threshold_, math.inf)
        self.assertEqual(router.route([0.9, 1.0]), [False, False])
        risk, coverage = router.evaluate([0.5] * 10, [True] * 10)
        self.assertEqual(coverage, 0.0)
        self.assertTrue(math.isnan(risk))

    def test_route_matches_confidence_against_the_threshold(self):
        router = ConformalRouter(target_risk=0.2).fit([0.9, 0.8, 0.3, 0.2], [True, True, False, False])
        accepted = router.route([0.85, 0.25])
        self.assertEqual(accepted, [router.threshold_ <= 0.85, router.threshold_ <= 0.25])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            ConformalRouter(target_risk=0.0)
        with self.assertRaises(ValueError):
            ConformalRouter(target_risk=1.0)
        with self.assertRaises(ValueError):
            ConformalRouter().fit([0.5], [True, False])
        with self.assertRaises(ValueError):
            ConformalRouter().fit([], [])
        with self.assertRaises(ValueError):
            ConformalRouter().fit([0.9, 0.8], [False, False])

    def test_routing_before_fit_is_an_error(self):
        with self.assertRaises(RuntimeError):
            ConformalRouter().route([0.9])

    def test_repr_reports_the_fitted_state(self):
        router, _, _ = calibrated_split(target_risk=0.05)
        self.assertIn("threshold=", repr(router))
        self.assertIn("unfitted", repr(ConformalRouter()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
