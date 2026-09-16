#!/usr/bin/env python3
"""Score a closed label set with a local model, one forward pass per batch.

    pip install "jev-rlcd-poc[local]"     # torch + transformers
    python examples/hf_single_pass.py --model Qwen/Qwen2.5-0.5B-Instruct

Every label's probability comes out of the same pass, so the cost per decision
is a batch cost divided by the batch size -- it does not grow with the number of
candidate labels the way generating and comparing strings would.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends.hf_local import LocalLabelScorer  # noqa: E402
from jev_rlcd_poc.decide import normalized_entropy, select  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

DEFAULT_LABELS = "billing,shipping,returns,account,other"

STATES = [
    "My invoice shows a charge I never authorized and I want it reversed.",
    "The package was supposed to arrive Tuesday and the tracking has not moved.",
    "I need to send back two pairs of shoes that do not fit.",
    "How do I change the email address on my profile?",
    "Can you tell me if you price match a competitor's sale next weekend?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="comma-separated, each must be one token")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    parser.add_argument("--system", default="Classify the customer message into exactly one category.")
    parser.add_argument("--prompt-template", default="{state}")
    args = parser.parse_args()

    label_set = LabelSet(tuple(part.strip() for part in args.labels.split(",") if part.strip()))
    scorer = LocalLabelScorer(
        label_set,
        model_name=args.model,
        system_prompt=args.system,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"{scorer}")
    print(f"label token ids: {scorer._label_token_ids}")
    print()

    scorer.warmup()  # burn the first-call overhead so timing is meaningful
    passes_before = scorer.forward_passes
    probs = scorer.score(STATES)

    for state, row in zip(STATES, probs):
        decision = select(row, label_set)
        ranking = sorted(zip(label_set.labels, row), key=lambda pair: -pair[1])
        spread = ", ".join(f"{label}={p:.3f}" for label, p in ranking)
        print(f"{state}")
        print(f"  -> {decision.label}  confidence={decision.confidence:.3f}  "
              f"entropy={normalized_entropy(row):.3f}")
        print(f"     {spread}")
    print()
    print(f"forward passes for {len(STATES)} decisions: "
          f"{scorer.forward_passes - passes_before} (batching {args.batch_size} at a time)")
    print(f"last batch wall clock: {scorer.last_batch_ms:.1f} ms total, "
          f"{scorer.last_ms_per_decision:.1f} ms per decision")
    print()
    print("No sampling, no parsing, and no way to return a label outside the set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
