"""Tests for the hosted logprobs backend.

The network is replaced by an injected transport, so these run offline and with
no API key. What is being tested is the parsing and the request shape -- the
parts that break silently when an endpoint changes.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends.openai_logprobs import (  # noqa: E402
    MissingLogprobError,
    OpenAICompatibleLogprobsScorer,
)
from jev_rlcd_poc.metrics import argmax, ece  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

LABELS = LabelSet(("billing", "shipping", "returns"))


def response(token_entries):
    """A canned chat-completions response with the given top_logprobs."""
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [{"token": token_entries[0][0], "logprobs": [None], "top_logprobs": [
                        {"token": token, "logprob": logprob} for token, logprob in token_entries
                    ]}]
                }
            }
        ]
    }


CANNED = response([(" billing", -0.1), (" shipping", -2.5), (" returns", -4.0), ("The", -1.0)])


class TestParsing(unittest.TestCase):
    def scorer(self, payload=CANNED, **kwargs):
        return OpenAICompatibleLogprobsScorer(
            LABELS, "some-model", transport=lambda request: payload, **kwargs
        )

    def test_row_follows_label_order_and_normalizes(self):
        row = self.scorer().score(["x"])[0]
        self.assertEqual(len(row), 3)
        self.assertAlmostEqual(sum(row), 1.0, places=9)
        self.assertEqual(argmax(row), 0)  # billing has the highest logprob

    def test_probabilities_are_the_softmax_of_the_returned_logprobs(self):
        row = self.scorer().score(["x"])[0]
        # Compare against the raw softmax of the three matched logprobs.
        expected = [math.exp(v) for v in (-0.1, -2.5, -4.0)]
        total = sum(expected)
        for got, want in zip(row, expected):
            self.assertAlmostEqual(got, want / total, places=9)

    def test_stripped_tokens_also_match(self):
        stripped = response([("billing", -0.1), ("shipping", -2.5), ("returns", -4.0)])
        row = self.scorer(payload=stripped).score(["x"])[0]
        self.assertEqual(argmax(row), 0)

    def test_missing_labels_get_the_floor_and_are_reported(self):
        payload = response([(" billing", -0.1), ("The", -0.5)])
        scorer = self.scorer(payload=payload)
        row = scorer.score(["x"])[0]
        self.assertEqual(scorer.last_missing, ["shipping", "returns"])
        self.assertEqual(argmax(row), 0)
        # The floor is far below the observed score, so the missing labels are
        # suppressed rather than treated as plausible.
        self.assertLess(row[1], 0.01)

    def test_all_labels_missing_is_an_error_with_a_fix(self):
        payload = response([("The", -0.1), ("Answer", -0.5)])
        with self.assertRaises(MissingLogprobError) as caught:
            self.scorer(payload=payload).score(["x"])
        self.assertIn("top_logprobs", str(caught.exception))

    def test_missing_logprobs_block_is_an_error(self):
        with self.assertRaises(MissingLogprobError):
            self.scorer(payload={"choices": [{"message": {"content": "hi"}}]}).score(["x"])

    def test_empty_content_is_an_error(self):
        payload = {"choices": [{"logprobs": {"content": []}}]}
        with self.assertRaises(MissingLogprobError):
            self.scorer(payload=payload).score(["x"])

    def test_requests_are_counted_one_per_state(self):
        scorer = self.scorer()
        scorer.score(["a", "b", "c"])
        self.assertEqual(scorer.requests, 3)


class TestRequestShape(unittest.TestCase):
    def test_payload_asks_for_one_token_and_the_logprobs(self):
        seen = {}

        def transport(payload):
            seen.update(payload)
            return CANNED

        scorer = OpenAICompatibleLogprobsScorer(
            LABELS,
            "gpt-x",
            system_prompt="Classify the message.",
            prompt_template="Message: {state}",
            top_logprobs=7,
            transport=transport,
        )
        scorer.score(["hello"])
        self.assertEqual(seen["model"], "gpt-x")
        self.assertEqual(seen["max_tokens"], 1)
        self.assertEqual(seen["temperature"], 0)
        self.assertTrue(seen["logprobs"])
        self.assertEqual(seen["top_logprobs"], 7)
        self.assertEqual(seen["messages"][0], {"role": "system", "content": "Classify the message."})
        self.assertEqual(seen["messages"][1]["content"], "Message: hello")

    def test_system_prompt_is_optional(self):
        seen = {}
        scorer = OpenAICompatibleLogprobsScorer(
            LABELS, "gpt-x", transport=lambda payload: (seen.update(payload), CANNED)[1]
        )
        scorer.score(["hello"])
        self.assertEqual(len(seen["messages"]), 1)

    def test_api_key_falls_back_to_the_environment(self):
        import os

        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "from-env"
        try:
            self.assertEqual(OpenAICompatibleLogprobsScorer(LABELS, "m").api_key, "from-env")
            self.assertEqual(
                OpenAICompatibleLogprobsScorer(LABELS, "m", api_key="explicit").api_key,
                "explicit",
            )
        finally:
            if previous is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = previous

    def test_rejects_invalid_top_logprobs(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleLogprobsScorer(LABELS, "m", top_logprobs=0)


class TestPipelineIntegration(unittest.TestCase):
    def test_hosted_scorer_feeds_the_same_pipeline(self):
        payloads = {
            "billing": response([(" billing", -0.05), (" shipping", -3.0), (" returns", -5.0)]),
            "shipping": response([(" shipping", -0.05), (" billing", -3.0), (" returns", -5.0)]),
            "returns": response([(" returns", -0.05), (" billing", -3.0), (" shipping", -5.0)]),
        }
        # One confident answer and one that mixes all three: ECE should see the
        # difference without any change to the calibration code.
        payloads["mixed"] = response([(" billing", -1.0), (" shipping", -1.1), (" returns", -1.2)])

        scorer = OpenAICompatibleLogprobsScorer(
            LABELS, "gpt-x", transport=lambda payload: payloads[payload["messages"][-1]["content"]]
        )
        states = ["billing", "shipping", "returns", "mixed"]
        y = [0, 1, 2, 0]
        probs = scorer.score(states)
        self.assertEqual([argmax(row) for row in probs], [0, 1, 2, 0])
        self.assertGreater(ece(probs, y), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
