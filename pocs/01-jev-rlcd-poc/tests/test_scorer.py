"""Tests for the decision interface: python -m unittest discover -s tests"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends import SyntheticDecisionTask  # noqa: E402
from jev_rlcd_poc.metrics import argmax, ece  # noqa: E402
from jev_rlcd_poc.scorer import (  # noqa: E402
    FunctionScorer,
    LabelSet,
    ScoredBatch,
    chunks,
    validate_prob_rows,
)


class TestLabelSet(unittest.TestCase):
    def test_basic_access(self):
        labels = LabelSet(("a", "b", "c"))
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[1], "b")
        self.assertEqual(labels.index_of("c"), 2)
        self.assertEqual(list(labels), ["a", "b", "c"])
        self.assertEqual(LabelSet.from_iterable(["x", "y"]).labels, ("x", "y"))

    def test_rejects_empty_and_duplicate_labels(self):
        with self.assertRaises(ValueError):
            LabelSet(())
        with self.assertRaises(ValueError):
            LabelSet(("a", "b", "a"))

    def test_unknown_label_raises_key_error(self):
        with self.assertRaises(KeyError):
            LabelSet(("a", "b")).index_of("z")

    def test_is_hashable_and_comparable(self):
        self.assertEqual(LabelSet(("a", "b")), LabelSet(("a", "b")))
        self.assertEqual(len({LabelSet(("a",)), LabelSet(("a",))}), 1)


class TestValidateProbRows(unittest.TestCase):
    labels = LabelSet(("a", "b"))

    def test_accepts_a_valid_batch(self):
        validate_prob_rows([[0.7, 0.3], [0.1, 0.9]], self.labels)

    def test_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            validate_prob_rows([[0.7, 0.2, 0.1]], self.labels)

    def test_rejects_unnormalized_rows(self):
        with self.assertRaises(ValueError):
            validate_prob_rows([[0.7, 0.7]], self.labels)

    def test_rejects_out_of_range_probabilities(self):
        with self.assertRaises(ValueError):
            validate_prob_rows([[1.4, -0.4]], self.labels)


class TestScoredBatch(unittest.TestCase):
    labels = LabelSet(("a", "b", "c"))

    def test_per_decision_cost_and_label_lookup(self):
        batch = ScoredBatch(
            states=["s1", "s2", "s3", "s4"],
            probs=[[0.1, 0.7, 0.2], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5], [0.1, 0.8, 0.1]],
            latency_ms=40.0,
            label_set=self.labels,
            forward_passes=1,
        )
        self.assertEqual(len(batch), 4)
        self.assertAlmostEqual(batch.ms_per_decision, 10.0)
        self.assertEqual(batch.labels(), ["b", "a", "c", "b"])
        self.assertEqual(batch.label_indices(), [1, 0, 2, 1])
        self.assertAlmostEqual(batch.confidences()[0], 0.7)

    def test_correctness_flags(self):
        batch = ScoredBatch(
            states=["s1", "s2"],
            probs=[[0.9, 0.1], [0.2, 0.8]],
            latency_ms=2.0,
            label_set=LabelSet(("a", "b")),
        )
        self.assertEqual(batch.correct([0, 0]), [True, False])
        with self.assertRaises(ValueError):
            batch.correct([0])


class TestFunctionScorer(unittest.TestCase):
    """The seam for using the harness with a hosted model or any other scorer."""

    labels = LabelSet(("billing", "shipping", "returns"))

    def assertRowAlmostEqual(self, row, expected):
        self.assertEqual(len(row), len(expected))
        for got, want in zip(row, expected):
            self.assertAlmostEqual(got, want, places=9)

    def test_mapping_input_is_read_in_label_order(self):
        scorer = FunctionScorer(
            self.labels, lambda state: {"returns": 0.1, "billing": 0.7, "shipping": 0.2}
        )
        self.assertRowAlmostEqual(scorer.score(["x"])[0], [0.7, 0.2, 0.1])

    def test_sequence_input_is_read_in_label_order(self):
        scorer = FunctionScorer(self.labels, lambda state: [0.7, 0.2, 0.1])
        self.assertRowAlmostEqual(scorer.score(["x"])[0], [0.7, 0.2, 0.1])

    def test_logprobs_are_converted_when_mode_is_logits(self):
        scorer = FunctionScorer(self.labels, lambda state: [-1.0, -2.0, -3.0], mode="logits")
        row = scorer.score(["x"])[0]
        self.assertAlmostEqual(sum(row), 1.0, places=9)
        self.assertEqual(argmax(row), 0)
        # Same softmax the local backend applies to raw label logits.
        self.assertAlmostEqual(row[1] / row[0], math.exp(-1.0), places=9)

    def test_negative_scores_without_logits_mode_name_the_fix(self):
        scorer = FunctionScorer(self.labels, lambda state: [-1.0, -2.0, -3.0])
        with self.assertRaises(ValueError) as caught:
            scorer.score(["x"])
        self.assertIn("mode='logits'", str(caught.exception))

    def test_rejects_an_unknown_mode(self):
        with self.assertRaises(ValueError):
            FunctionScorer(self.labels, lambda state: [1, 0, 0], mode="logprob")

    def test_unnormalized_rows_are_rejected_when_normalization_is_off(self):
        scorer = FunctionScorer(self.labels, lambda state: [2.0, 2.0, 2.0], normalize=False)
        with self.assertRaises(ValueError) as caught:
            scorer.score(["x"])
        self.assertIn("renormalize", str(caught.exception))

    def test_missing_label_is_reported(self):
        scorer = FunctionScorer(self.labels, lambda state: {"billing": 0.5, "shipping": 0.5})
        with self.assertRaises(ValueError) as caught:
            scorer.score(["state-7"])
        self.assertIn("returns", str(caught.exception))
        self.assertIn("state-7", str(caught.exception))

    def test_wrong_length_is_reported(self):
        scorer = FunctionScorer(self.labels, lambda state: [0.5, 0.5])
        with self.assertRaises(ValueError):
            scorer.score(["x"])

    def test_negative_and_empty_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            FunctionScorer(self.labels, lambda state: [1.5, -0.5, 0.0]).score(["x"])
        with self.assertRaises(ValueError):
            FunctionScorer(self.labels, lambda state: [0.0, 0.0, 0.0]).score(["x"])

    def test_plugs_into_the_metrics_unchanged(self):
        scorers = {
            "billing": [0.9, 0.05, 0.05],
            "shipping": [0.05, 0.9, 0.05],
            "returns": [0.05, 0.05, 0.9],
            "account": [0.6, 0.3, 0.1],
        }
        scorer = FunctionScorer(self.labels, lambda state: scorers[state])
        states = ["billing", "shipping", "returns", "account"]
        y = [0, 1, 2, 0]
        probs = scorer.score(states)
        # All four are right, but one only claims 0.6. ECE is the sample-weighted
        # gap per bin: (3/4) * (0.9 - 1.0) + (1/4) * (0.6 - 1.0) = 0.175.
        self.assertAlmostEqual(ece(probs, y), 0.175, places=9)


class TestChunks(unittest.TestCase):
    def test_splits_into_trailing_partial_batch(self):
        self.assertEqual(list(chunks([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(chunks([], 2)), [])

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            list(chunks([1], 0))


class TestSyntheticBackend(unittest.TestCase):
    def test_realized_accuracy_matches_the_requested_accuracy(self):
        for accuracy in (0.5, 0.72, 0.9):
            task = SyntheticDecisionTask(n_labels=16, accuracy=accuracy, seed=2)
            states, y = task.sample(4000, seed=3)
            probs = task.scorer().score(states)
            realized = sum(1 for row, target in zip(probs, y) if argmax(row) == target) / len(y)
            self.assertAlmostEqual(realized, accuracy, delta=0.03)

    def test_the_model_is_actually_overconfident(self):
        task = SyntheticDecisionTask(n_labels=16, accuracy=0.7, overconfidence=8.0, seed=4)
        states, y = task.sample(3000, seed=5)
        probs = task.scorer().score(states)
        correct = [argmax(row) == target for row, target in zip(probs, y)]
        # Among the decisions it claims to be nearly certain about, how much of
        # that certainty is real?
        confident = [(max(row), ok) for row, ok in zip(probs, correct) if max(row) > 0.9]
        self.assertGreater(len(confident), 0)
        stated = sum(c for c, _ in confident) / len(confident)
        realized = sum(1 for _, ok in confident if ok) / len(confident)
        self.assertGreater(stated - realized, 0.1)

    def test_scoring_is_deterministic(self):
        task = SyntheticDecisionTask(seed=6)
        states, _ = task.sample(50, seed=7)
        self.assertEqual(task.scorer().score(states), task.scorer().score(states))

    def test_unknown_state_is_rejected(self):
        task = SyntheticDecisionTask(seed=8)
        with self.assertRaises(KeyError):
            task.scorer().score(["never-generated"])

    def test_labels_are_unique_per_case_and_tracked(self):
        task = SyntheticDecisionTask(seed=9)
        states, y = task.sample(500, seed=10)
        self.assertEqual(len(set(states)), 500)
        mismatches = sum(1 for state, target in zip(states, y) if task.truth[state] != target)
        self.assertEqual(mismatches, 0)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            SyntheticDecisionTask(n_labels=1)
        with self.assertRaises(ValueError):
            SyntheticDecisionTask(accuracy=1.0)
        with self.assertRaises(ValueError):
            SyntheticDecisionTask(overconfidence=0.0)

    def test_sampling_needs_a_positive_count(self):
        with self.assertRaises(ValueError):
            SyntheticDecisionTask().sample(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
