"""Scorer backends.

``synthetic`` is dependency-free and is what the tests and the demo report use:
it simulates a model that is decent but overconfident, so the calibration and
risk-coverage machinery can be exercised without downloading anything.

``openai_logprobs`` talks to any OpenAI-compatible endpoint, using only the
standard library, so a hosted scorer adds no dependency.

``hf_local`` needs ``pip install "jev-rlcd-poc[local]"`` and is imported
explicitly (``from jev_rlcd_poc.backends.hf_local import LocalLabelScorer``) so that
torch stays optional.
"""

from .openai_logprobs import MissingLogprobError, OpenAICompatibleLogprobsScorer
from .synthetic import SyntheticDecisionTask, SyntheticScorer

__all__ = [
    "MissingLogprobError",
    "OpenAICompatibleLogprobsScorer",
    "SyntheticDecisionTask",
    "SyntheticScorer",
]
