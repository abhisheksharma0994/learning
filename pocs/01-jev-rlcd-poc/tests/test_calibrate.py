"""Tests for post-hoc calibration: python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc._mathx import softmax  # noqa: E402
from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.calibrate import HistogramBinner, TemperatureScaler  # noqa: E402
from jev_rlcd_poc.metrics import argmax, ece, nll, top1_accuracy  # noqa: E402


def build_dataset(n_cal: int = 4000, n_test: int = 4000, **task_kwargs):
    task = SyntheticDecisionTask(seed=11, **task_kwargs)
    cal_states, cal_y = task.sample(n_cal, seed=1)
    test_states, test_y = task.sample(n_test, seed=2)
    scorer = task.scorer()
    return task, scorer.score(cal_states), cal_y, scorer.score(test_states), test_y


class TestTemperatureScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.task,
            cls.cal_probs,
            cls.cal_y,
            cls.test_probs,
            cls.test_y,
        ) = build_dataset(accuracy=0.7, overconfidence=8.0, winner_bias=0.0)

    def test_temperature_is_above_one_for_overconfident_data(self):
        scaler = TemperatureScaler().fit(self.cal_probs, self.cal_y)
        self.assertGreater(scaler.temperature, 1.0)

    def test_reduces_ece_and_nll_on_held_out_data(self):
        scaler = TemperatureScaler().fit(self.cal_probs, self.cal_y)
        calibrated = scaler.transform(self.test_probs)
        self.assertLess(ece(calibrated, self.test_y), ece(self.test_probs, self.test_y))
        self.assertLess(nll(calibrated, self.test_y), nll(self.test_probs, self.test_y))

    def test_preserves_predictions_and_accuracy(self):
        scaler = TemperatureScaler().fit(self.cal_probs, self.cal_y)
        calibrated = scaler.transform(self.test_probs)
        # Compare a mismatch count rather than the lists themselves: unittest
        # would otherwise run a quadratic difflib diff over thousands of ints.
        mismatches = sum(
            1
            for before, after in zip(self.test_probs, calibrated)
            if argmax(before) != argmax(after)
        )
        self.assertEqual(mismatches, 0)
        self.assertAlmostEqual(
            top1_accuracy(calibrated, self.test_y),
            top1_accuracy(self.test_probs, self.test_y),
        )

    def test_fitting_on_logits_matches_fitting_on_their_softmax(self):
        # The equivalence that lets one calibrator serve both code paths: a
        # softmax output and its logits differ only by a per-row constant, and
        # softmax ignores that, so dividing by T is the same operation.
        task = SyntheticDecisionTask(seed=3, overconfidence=1.8)
        states, y = task.sample(1500, seed=4)
        logits = task.scorer().score_logits(states)
        probs = [softmax(row) for row in logits]
        from_probs = TemperatureScaler().fit(probs, y).temperature
        from_logits = TemperatureScaler().fit_from_logits(logits, y).temperature
        self.assertAlmostEqual(from_probs, from_logits, places=6)

    def test_softmax_output_rows_stay_normalized(self):
        scaler = TemperatureScaler().fit(self.cal_probs, self.cal_y)
        for row in scaler.transform(self.test_probs[:20]):
            self.assertAlmostEqual(sum(row), 1.0, places=9)

    def test_transform_before_fit_is_an_error(self):
        with self.assertRaises(RuntimeError):
            TemperatureScaler().transform([[0.5, 0.5]])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            TemperatureScaler(temperature=0.0)
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([[0.5, 0.5]], [0, 1])  # length mismatch
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([[0.5, 0.5]], [5])  # label out of range
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([[0.5, 0.5]], [0], search_range=(10.0, 1.0))


class TestHistogramBinner(unittest.TestCase):
    def test_reduces_ece_on_held_out_data(self):
        _, cal_probs, cal_y, test_probs, test_y = build_dataset(accuracy=0.7)
        binner = HistogramBinner(n_bins=12).fit(cal_probs, cal_y)
        calibrated = binner.transform(test_probs)
        self.assertLess(ece(calibrated, test_y), ece(test_probs, test_y))

    def test_preserves_predictions(self):
        _, cal_probs, cal_y, test_probs, test_y = build_dataset(accuracy=0.7)
        binner = HistogramBinner(n_bins=12).fit(cal_probs, cal_y)
        calibrated = binner.transform(test_probs)
        mismatches = sum(
            1
            for before, after in zip(test_probs, calibrated)
            if argmax(before) != argmax(after)
        )
        self.assertEqual(mismatches, 0)

    def test_argmax_survives_a_row_that_must_be_softened(self):
        # Stated 0.45 on the top label, runner-up at 0.40, but the bin's realized
        # accuracy is 0.38. Naively setting the top to 0.38 and scaling the tail
        # up proportionally would push the runner-up to 0.45 and flip the answer.
        probs = [[0.45, 0.40, 0.15]] * 100
        y = [0] * 38 + [1] * 62
        binner = HistogramBinner(n_bins=1).fit(probs, y)
        calibrated = binner.transform(probs[:5])
        for row in calibrated:
            self.assertEqual(argmax(row), 0)
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            self.assertAlmostEqual(max(row), 0.38, places=5)

    def test_row_is_untouched_when_the_target_is_below_uniform(self):
        probs = [[0.5, 0.3, 0.2]] * 10
        y = [1] * 10  # bin accuracy 0.0, unrepresentable while keeping the ranking
        binner = HistogramBinner(n_bins=1).fit(probs, y)
        self.assertEqual(binner.transform(probs[:2]), probs[:2])

    def test_bin_accuracies_are_monotone_by_construction(self):
        _, cal_probs, cal_y, _, _ = build_dataset(accuracy=0.7)
        binner = HistogramBinner(n_bins=15).fit(cal_probs, cal_y)
        self.assertEqual(binner.accuracies_, sorted(binner.accuracies_))
        self.assertEqual(len(binner.edges_), len(binner.accuracies_))
        self.assertEqual(sum(binner.counts_), len(cal_probs))

    def test_handles_more_bins_than_examples(self):
        probs = [[0.7, 0.3], [0.6, 0.4]]
        y = [0, 1]
        binner = HistogramBinner(n_bins=50).fit(probs, y)
        self.assertEqual(len(binner.edges_), 2)
        self.assertEqual(len(binner.transform(probs)), 2)

    def test_transform_before_fit_is_an_error(self):
        with self.assertRaises(RuntimeError):
            HistogramBinner().transform([[0.5, 0.5]])

    def test_rejects_invalid_bin_count(self):
        with self.assertRaises(ValueError):
            HistogramBinner(n_bins=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
