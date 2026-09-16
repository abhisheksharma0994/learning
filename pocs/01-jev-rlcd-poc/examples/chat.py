#!/usr/bin/env python3
"""A chat loop whose control flow is made of calibrated decisions.

Two roles, deliberately separated:

* the **language model** writes the open-ended prose -- the part that is
  unavoidably free-form;
* the **calibrated scorer** makes the closed-set decision, and reports a
  confidence you can act on, so the loop can escalate instead of bluffing.

One decision drives all three code paths. The label set contains ``chat``, so the
same forward pass answers "is this even a support request?" and "which team owns
it?" -- and the calibrated confidence decides whether to route automatically or
hand off to a human.

Read the calibration report before judging the bot. An off-the-shelf 0.5B chat
model is a *weak* label scorer on six teams (about 60% raw accuracy here, and it
barely ever emits ``other``), and its raw confidence is badly overconfident. The
point of the report is that the harness measures this and gates on it: the loop
auto-routes only the slice it can justify and escalates the rest. Swap the scorer
(``--model``, or the ``FunctionScorer`` shown in the README) and the same code
becomes useful rather than merely safe.

    python examples/chat.py --report                 # calibration on held-out data
    python examples/chat.py --say "my invoice looks wrong"
    python examples/chat.py                          # interactive

Needs the local extra: pip install "jev-rlcd-poc[local]"
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends.hf_local import LocalLabelScorer  # noqa: E402
from jev_rlcd_poc.calibrate import TemperatureScaler  # noqa: E402
from jev_rlcd_poc.decide import ConformalRouter, select  # noqa: E402
from jev_rlcd_poc.llm import LocalChatModel  # noqa: E402
from jev_rlcd_poc.metrics import argmax, ece, top1_accuracy  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

# Every label must be a single token for the single-pass scorer, which is why
# this one is "chat" rather than "smalltalk" -- the backend rejects the longer
# spelling instead of silently falling back to sequential decoding.
LABELS = LabelSet(("billing", "shipping", "returns", "account", "other", "chat"))
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Cues are chosen so the correct label is unambiguous: this is the ground truth
# the confidence gets measured against. Swap this section, and the data loader,
# for your own labelled set and nothing below changes.
INTENT_CUES = {
    "billing": [
        "an unexpected invoice charge",
        "my payment failing at checkout",
        "a billing statement I do not recognise",
        "the subscription price changing",
        "a duplicate charge on my card",
        "a refund for a double payment",
    ],
    "shipping": [
        "a delivery that never arrived",
        "tracking that has not updated",
        "a package sent to the wrong address",
        "an order that shipped late",
        "the courier leaving a note instead of delivering",
        "a delivery window I need to change",
    ],
    "returns": [
        "returning a faulty item",
        "an exchange for a different size",
        "a return label that will not print",
        "sending back an unwanted gift",
        "a restocking fee on a return",
        "an item that arrived damaged",
    ],
    "account": [
        "resetting my password",
        "logging in from a new device",
        "changing my email address",
        "two-factor authentication not sending codes",
        "closing my account",
        "updating my profile details",
    ],
    "other": [
        "a price match request",
        "a partnership proposal",
        "your careers page",
        "a press enquiry",
        "general feedback about the website",
        "whether you serve customers in other countries",
    ],
}

SUPPORT_TEMPLATES = [
    "I need help with {cue}.",
    "Can you look into {cue} for me?",
    "There is a problem with {cue}.",
    "Please advise on {cue}.",
    "Hello, I have a question about {cue}.",
    "{cue} is not working as expected.",
    "I would like to raise an issue about {cue}.",
    "Following up on {cue} from last week.",
    "I am writing about {cue} and need a hand.",
    "Could someone help me resolve {cue}?",
    "Question regarding {cue} please.",
    "I have been having trouble with {cue} since yesterday.",
]

SUFFIXES = ["", "", " Thanks.", " Please help.", " This is urgent.", " Any update?"]

SMALL_TALK = [
    "hey there",
    "hello!",
    "good morning",
    "thanks so much",
    "who are you?",
    "are you a robot?",
    "haha nice",
    "ok cool",
    "no thanks, just browsing",
    "have a good one",
    "what can you do?",
    "bye for now",
    "just testing this chat",
    "you there?",
    "nice weather today",
    "I like this new layout",
    "hello again",
    "cheers",
]

SMALL_TALK_VARIANTS = ["{}", "{}.", "so {}", "{} :)", "oh, {}", "hi - {}"]


def build_dataset(n_support: int = 360, n_smalltalk: int = 108):
    """Return ``(states, targets)`` where targets index ``LABELS``.

    Nothing here is machine-generated prose with an invented answer: every label
    is determined by the cue the message is built from, so the ground truth is
    exact and the calibration below is measuring something real.
    """
    rng = random.Random(0)
    states: list[str] = []
    targets: list[int] = []

    support: list[tuple[str, int]] = []
    for intent, cues in INTENT_CUES.items():
        target = LABELS.index_of(intent)
        for cue in cues:
            for template in SUPPORT_TEMPLATES:
                support.append((template.format(cue=cue), target))
    rng.shuffle(support)
    for message, target in support[:n_support]:
        states.append(message + rng.choice(SUFFIXES))
        targets.append(target)

    smalltalk_target = LABELS.index_of("chat")
    chatter = [
        variant.format(phrase)
        for phrase in SMALL_TALK
        for variant in SMALL_TALK_VARIANTS
    ]
    rng.shuffle(chatter)
    for message in chatter[:n_smalltalk]:
        states.append(message)
        targets.append(smalltalk_target)

    # A calibration set drawn from one end of the list and a test set from the
    # other keeps the two disjoint without needing a second generator.
    order = list(range(len(states)))
    rng.shuffle(order)
    states = [states[i] for i in order]
    targets = [targets[i] for i in order]
    return states, targets


class DecisionEngine:
    """A calibrated scorer, plus the threshold that makes its confidence usable."""

    def __init__(self, label_set, scorer, cal_probs, cal_y, test_probs, test_y, target_risk):
        self.label_set = label_set
        self.scorer = scorer
        self.scaler = TemperatureScaler().fit(cal_probs, cal_y)
        self.n_cal = len(cal_probs)
        self.n_test = len(test_probs)
        self.test_probs = test_probs
        self.test_y = test_y
        self.raw_ece = ece(test_probs, test_y)
        self.cal_ece = ece(self.scaler.transform(test_probs), test_y)
        self.accuracy = top1_accuracy(test_probs, test_y)

        cal_calibrated = self.scaler.transform(cal_probs)
        cal_conf = [max(row) for row in cal_calibrated]
        cal_correct = [argmax(row) == y for row, y in zip(cal_calibrated, cal_y)]
        self.router = ConformalRouter(target_risk=target_risk).fit(cal_conf, cal_correct)

        test_calibrated = self.scaler.transform(test_probs)
        self.risk, self.coverage = self.router.evaluate(
            [max(row) for row in test_calibrated],
            [argmax(row) == y for row, y in zip(test_calibrated, test_y)],
        )

    def decide(self, state: str):
        """Return ``(decision, probabilities, accepted)``."""
        raw = self.scorer.score([state])[0]
        probs = self.scaler.transform([raw])[0]
        decision = select(probs, self.label_set)
        accepted = self.router.route([decision.confidence])[0]
        return decision, probs, accepted

    def per_label_rows(self) -> list[list[str]]:
        """Where the model is competent, and whether its confidence knows that.

        The useful column is the last one. A label the model cannot do should
        also be a label whose confidence sits below the routing threshold; if
        those two columns disagree, the gate is not protecting you.
        """
        rows = []
        for k, name in enumerate(self.label_set.labels):
            selected = [i for i, target in enumerate(self.test_y) if target == k]
            if not selected:
                continue
            calibrated = self.scaler.transform([self.test_probs[i] for i in selected])
            accuracy = sum(
                1 for i in selected if argmax(self.test_probs[i]) == k
            ) / len(selected)
            mean_confidence = sum(max(row) for row in calibrated) / len(calibrated)
            accepted = sum(
                1 for row in calibrated if self.router.route([max(row)])[0]
            ) / len(calibrated)
            rows.append(
                [
                    name,
                    str(len(selected)),
                    f"{accuracy:.2f}",
                    f"{mean_confidence:.3f}",
                    f"{accepted:.0%}",
                ]
            )
        return rows

    def report(self) -> list[str]:
        threshold = "unreachable" if not self.router.reachable else f"{self.router.threshold_:.3f}"
        risk = "n/a" if self.risk != self.risk else f"{self.risk:.1%}"
        return [
            f"  labelled decisions          {self.n_cal} calibration / {self.n_test} held out",
            f"  accuracy on held-out data   {self.accuracy:.3f}",
            f"  ECE raw -> calibrated       {self.raw_ece:.4f} -> {self.cal_ece:.4f}",
            f"  accept if confidence >=     {threshold}",
            f"  held-out coverage / risk    {self.coverage:.1%} auto-routed at {risk} error",
        ]


class Chat:
    def __init__(self, model_name: str, target_risk: float, device: str | None = None):
        self.scorer = LocalLabelScorer(
            LABELS,
            model_name=model_name,
            system_prompt="Classify the customer message into exactly one team.",
            device=device,
        )
        self.llm = LocalChatModel(model_name, device=device, max_new_tokens=96)

        states, targets = build_dataset()
        split = len(states) // 2
        self.engine = DecisionEngine(
            LABELS,
            self.scorer,
            self.scorer.score(states[:split]),
            targets[:split],
            self.scorer.score(states[split:]),
            targets[split:],
            target_risk,
        )

    def report(self) -> None:
        print("Calibration, measured on data the calibrators never saw")
        print()
        print("one decision: which team owns this message, if any")
        print("\n".join(self.engine.report()))
        print()

    def diagnostics(self) -> None:
        print("Per label, on held-out data")
        print()
        print_table(
            ["label", "n", "acc", "calibrated conf", "auto-accepted"],
            self.engine.per_label_rows(),
        )
        print()
        print("If the model is bad at a label and its confidence there is also low,")
        print("the gate is doing its job: those rows escalate instead of shipping.")
        print()

    def turn(self, message: str) -> None:
        decision, probs, accepted = self.engine.decide(message)
        ranking = sorted(zip(LABELS.labels, probs), key=lambda pair: -pair[1])[:3]
        spread = ", ".join(f"{label} {p:.3f}" for label, p in ranking)
        print(f"\nyou  > {message}")
        print(
            f"     [decision] {decision.label} p={decision.confidence:.3f} "
            f"({'accept' if accepted else 'ABSTAIN'})"
        )
        print(f"     [top 3]    {spread}")

        if not accepted:
            print(
                "bot  > I am not confident enough to route this on my own, so a human "
                "should take it. The distribution above is my best guess."
            )
            return

        if decision.label == "chat":
            system = (
                "You are a friendly, concise support assistant. Reply in one or two "
                "sentences and stay on topic."
            )
            print(f"bot  > {self.llm.reply(message, system)}")
            return

        print(
            f"bot  > Passing this to the {decision.label} team "
            f"(confidence {decision.confidence:.0%})."
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    parser.add_argument("--target-risk", type=float, default=0.10, help="error rate to accept when auto-routing")
    parser.add_argument("--report", action="store_true", help="print the calibration report and exit")
    parser.add_argument("--diagnostics", action="store_true", help="add the per-label competence table")
    parser.add_argument("--say", default=None, help="answer one message and exit")
    args = parser.parse_args()

    if args.report or args.say or args.diagnostics:
        chat = Chat(args.model, args.target_risk, device=args.device)
        chat.report()
        if args.diagnostics:
            chat.diagnostics()
        if args.say:
            chat.turn(args.say)
        return 0

    chat = Chat(args.model, args.target_risk, device=args.device)
    print("Chat with a calibrated decision engine. Type 'quit' to leave.\n")
    chat.report()
    while True:
        try:
            message = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"quit", "exit", "q"}:
            return 0
        if message:
            chat.turn(message)


if __name__ == "__main__":
    raise SystemExit(main())
