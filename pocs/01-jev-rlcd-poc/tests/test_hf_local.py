"""Smoke test for the local single-pass backend.

Downloading weights and running a forward pass is too heavy for the default
suite, so this is opt-in:

    JEV_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m unittest discover -s tests -p "test_hf_local.py"

Everything else in tests/ runs with no dependencies at all.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.metrics import argmax  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet, validate_prob_rows  # noqa: E402

MODEL = os.environ.get("JEV_TEST_MODEL")


@unittest.skipUnless(MODEL, "set JEV_TEST_MODEL to run the local-model smoke test")
class TestLocalLabelScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jev_rlcd_poc.backends.hf_local import LocalLabelScorer

        cls.labels = LabelSet(("billing", "shipping", "returns", "account", "other"))
        cls.scorer = LocalLabelScorer(cls.labels, model_name=MODEL, batch_size=4)
        cls.states = [
            "The package was supposed to arrive Tuesday and tracking has not moved.",
            "I need to send back two pairs of shoes that do not fit.",
        ]
        cls.probs = cls.scorer.score(cls.states)

    def test_scores_every_label_in_one_row(self):
        validate_prob_rows(self.probs, self.labels)
        self.assertEqual(len(self.probs), len(self.states))

    def test_one_forward_pass_per_batch(self):
        # The whole point: scoring 5 labels costs the same single pass as 1.
        self.assertEqual(self.scorer.forward_passes, 1)

    def test_labels_resolve_to_single_tokens(self):
        self.assertEqual(len(self.scorer._label_token_ids), len(self.labels))
        self.assertEqual(len(set(self.scorer._label_token_ids)), len(self.labels))

    def test_predictions_are_in_the_label_set(self):
        for state, row in zip(self.states, self.probs):
            self.assertIn(self.labels[argmax(row)], self.labels.labels)


class TestLabelRejection(unittest.TestCase):
    def test_multi_token_labels_are_refused(self):
        if not MODEL:
            self.skipTest("set JEV_TEST_MODEL to run")
        from jev_rlcd_poc.backends.hf_local import LocalLabelScorer

        labels = LabelSet(("this label is far too long to be one token", "billing"))
        with self.assertRaises(ValueError) as caught:
            LocalLabelScorer(labels, model_name=MODEL)
        self.assertIn("single token", str(caught.exception))

    def test_torch_is_only_required_for_this_backend(self):
        # The rest of the package imports without torch installed.
        if not MODEL:
            self.skipTest("set JEV_TEST_MODEL to run")
        import jev_rlcd_poc

        self.assertTrue(hasattr(jev_rlcd_poc, "TemperatureScaler"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
