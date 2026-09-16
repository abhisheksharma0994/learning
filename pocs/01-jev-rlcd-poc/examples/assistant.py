#!/usr/bin/env python3
"""A general chat assistant with its decision made in one pass.

What actually gets faster, stated precisely: **the decision, not the prose.**
Generation costs one forward pass per token; a decision costs exactly one pass,
no matter how many labels are in play. So the loop spends a pass deciding what
kind of question it is looking at, and generates only the answer.

Per turn:

1. ``route`` (one pass, calibrated): general knowledge or arithmetic?
2. **act**: arithmetic goes to a real evaluator, so the number is exact and
   generation never touches it. Otherwise the model answers in prose.
3. If the calibrated confidence is below the fitted threshold, the assistant asks
   for clarification instead of guessing.

Two findings from building this, both measured on held-out data with
Qwen2.5-1.5B-Instruct:

* **Classify the input, do not ask for an action.** ``("answer", "calculate")``
  with "choose whether this needs a calculator" scored **0.670**; the same data
  framed as ``("general", "arithmetic")`` with "classify the question" scored
  **0.993**. The model is far better at naming what a text *is* than at deciding
  what it should *do*. The action is then a lookup in code, which is the whole
  point of typed decisions.
* **A draft-utility check did not work, so there isn't one.** A second pass
  judging "is this draft responsive, or does it dodge?" collapsed to a single
  class in every framing I tried (0.63-0.79 accuracy at 0.95+ confidence, with
  one class recalled at 0.00). Reported here rather than shipped, because a
  guardrail that always says the same thing is worse than no guardrail.

    python examples/assistant.py --report
    python examples/assistant.py --say "what is 47 * 12"
    python examples/assistant.py --bench             # measured latency + exactness
    python examples/assistant.py --bench --no-tool   # same turns, tool disabled
    python examples/assistant.py                     # interactive

Needs the local extra: pip install "jev-rlcd-poc[local]"
"""

from __future__ import annotations

import argparse
import ast
import operator
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends.hf_local import LocalLabelScorer  # noqa: E402
from jev_rlcd_poc.calibrated import CalibratedDecider  # noqa: E402
from jev_rlcd_poc.llm import LocalChatModel  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

#: Content labels, not action labels. "arithmetic" describes the question;
#: "calculate" would describe what the model should do, and it does that worse.
ROUTE_LABELS = LabelSet(("general", "arithmetic"))
ROUTE_PROMPT = "Classify the question."
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ANSWER_SYSTEM = (
    "You are a concise, honest assistant. Answer in one or two sentences. "
    "If you do not know, say so plainly."
)

# -----------------------------------------------------------------------------
# The one tool. Deterministic, so nothing downstream has to trust generation.
# -----------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_EXPRESSION = re.compile(r"\d[\d\s.+\-*/()%]*\d|\d")


def calculate(expression: str) -> float:
    """Evaluate a pure arithmetic expression. No names, no calls, no imports."""
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError:
        raise ValueError(f"not an arithmetic expression: {expression!r}") from None

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError("only numbers are allowed")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 32:
                raise ValueError("exponent too large")  # 9**9**9 would never finish
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return evaluate(tree)


def find_expression(message: str) -> str | None:
    match = _EXPRESSION.search(message)
    if not match:
        return None
    candidate = match.group(0).strip()
    return candidate if any(op in candidate for op in "+-*/%") else None


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


# -----------------------------------------------------------------------------
# Labelled calibration data. Every label follows from the template used, so the
# confidence below is measured against an exact ground truth.
# -----------------------------------------------------------------------------

ARITHMETIC_TEMPLATES = [
    "what is {a} {op} {b}",
    "{a} {op} {b} = ?",
    "please compute {a} {op} {b}",
    "what does {a} {op} {b} equal",
    "how much is {a} {op} {b}",
]
KNOWLEDGE_TEMPLATES = [
    "who wrote {subject}",
    "what is the capital of {subject}",
    "explain {subject} in one sentence",
    "what does {subject} stand for",
    "why does {subject} matter",
]
SUBJECTS = [
    "Hamlet", "France", "Japan", "recursion", "HTTP", "TCP", "a leap year",
    "the water cycle", "Portugal", "inflation", "the moon", "an API", "Peru",
]


def build_route_data(n_arithmetic: int = 200, n_knowledge: int = 200):
    """Return ``(states, targets)`` with targets indexing ``ROUTE_LABELS``."""
    rng = random.Random(1)
    states, targets = [], []
    for _ in range(n_arithmetic):
        a, b = rng.randrange(11, 999), rng.randrange(2, 99)
        op = rng.choice(["+", "-", "*", "/"])
        states.append(rng.choice(ARITHMETIC_TEMPLATES).format(a=a, b=b, op=op))
        targets.append(ROUTE_LABELS.index_of("arithmetic"))
    for _ in range(n_knowledge):
        states.append(rng.choice(KNOWLEDGE_TEMPLATES).format(subject=rng.choice(SUBJECTS)))
        targets.append(ROUTE_LABELS.index_of("general"))
    order = list(range(len(states)))
    rng.shuffle(order)
    return [states[i] for i in order], [targets[i] for i in order]


@dataclass
class TurnResult:
    message: str
    label: str
    confidence: float
    accepted: bool
    decision_ms: float
    used_tool: bool
    expression: str | None
    answer: str
    generation_ms: float
    generated_tokens: int
    total_ms: float
    expected: float | None = None

    @property
    def correct(self) -> bool | None:
        """For arithmetic turns: does the answer contain the exact value?"""
        if self.expected is None:
            return None
        return format_number(self.expected) in self.answer


class Assistant:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        target_risk: float = 0.10,
        use_tool: bool = True,
        decision_scorer: str = "local",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        if decision_scorer == "hosted":
            # The swap: the same label set, the same pipeline, probabilities now
            # read from a hosted model's logprobs instead of local weights.
            from jev_rlcd_poc.backends.openai_logprobs import OpenAICompatibleLogprobsScorer

            self.scorer = OpenAICompatibleLogprobsScorer(
                ROUTE_LABELS,
                model=model_name,
                base_url=base_url,
                system_prompt=ROUTE_PROMPT,
            )
        else:
            self.scorer = LocalLabelScorer(
                ROUTE_LABELS,
                model_name=model_name,
                system_prompt=ROUTE_PROMPT,
                device=device,
            )

        self.llm = LocalChatModel(model_name, device=device, max_new_tokens=80)
        self.use_tool = use_tool
        # One throwaway pass, so the first measured decision is not paying for
        # lazy device initialisation. Latency claims should be honest ones.
        warmup = getattr(self.scorer, "warmup", None)
        if callable(warmup):
            warmup()

        states, targets = build_route_data()
        split = len(states) // 2
        self.decider = CalibratedDecider.fit(
            self.scorer, states[:split], targets[:split], target_risk=target_risk
        )
        self.report_data = self.decider.evaluate(states[split:], targets[split:])

    def report(self) -> None:
        print("Calibration on held-out data: what kind of question is this")
        print()
        print("\n".join("  " + line for line in self.report_data.lines()))
        print()

    def turn(self, message: str, max_new_tokens: int | None = None) -> TurnResult:
        started = time.perf_counter()

        # One forward pass. No tokens generated for the decision itself.
        decision_started = time.perf_counter()
        decision, probs, accepted = self.decider.decide(message)
        decision_ms = (time.perf_counter() - decision_started) * 1000.0

        # The action is a lookup on the label, not a second model call.
        expression = find_expression(message)
        used_tool = self.use_tool and accepted and decision.label == "arithmetic" and expression

        if used_tool:
            value = calculate(expression)
            # The number is placed by code and never regenerated by the model.
            answer = f"{expression} = {format_number(value)}."
            generation_ms, generated_tokens = 0.0, 0
        else:
            answer = self.llm.reply(message, ANSWER_SYSTEM, max_new_tokens=max_new_tokens)
            generation_ms, generated_tokens = self.llm.last_latency_ms, self.llm.last_tokens

        total_ms = (time.perf_counter() - started) * 1000.0
        expected = None
        if expression:
            try:
                expected = calculate(expression)
            except ValueError:
                expected = None
        return TurnResult(
            message=message,
            label=decision.label,
            confidence=decision.confidence,
            accepted=accepted,
            decision_ms=decision_ms,
            used_tool=bool(used_tool),
            expression=expression,
            answer=answer,
            generation_ms=generation_ms,
            generated_tokens=generated_tokens,
            total_ms=total_ms,
            expected=expected,
        )

    def show(self, result: TurnResult) -> None:
        print(f"\nyou  > {result.message}")
        print(
            f"     [decide] {result.label} p={result.confidence:.3f} "
            f"({result.decision_ms:.0f} ms, one pass"
            + (", tool used" if result.used_tool else "")
            + ")"
        )
        if not result.accepted:
            print(
                "bot  > I am not confident enough to pick an approach here, so rather "
                "than guess: could you rephrase what you need?"
            )
            return
        if result.generation_ms:
            print(
                f"     [answer] {result.generated_tokens} tokens in "
                f"{result.generation_ms:.0f} ms (generated)"
            )
        print(f"bot  > {result.answer}")


BENCH_TURNS = [
    ("what is 47 * 12", 564),
    ("what is 128 + 367", 495),
    ("what is 1443 - 869", 574),
    ("what is 96 / 8", 12),
    ("what is 17 * 23", 391),
    ("what is 827 * 34", 28118),
    ("who wrote Hamlet", None),
    ("what is the capital of Peru", None),
    ("what does HTTP stand for", None),
    ("explain recursion in one sentence", None),
    ("why does the moon affect tides", None),
    ("what is the capital of Portugal", None),
]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def bench(assistant: Assistant) -> int:
    results = [assistant.turn(message) for message, _ in BENCH_TURNS]
    for result in results:
        assistant.show(result)

    print()
    print("per turn")
    print_table(
        ["turn", "label", "p", "decide ms", "gen tokens", "gen ms", "arithmetic"],
        [
            [
                r.message[:32],
                r.label,
                f"{r.confidence:.2f}",
                f"{r.decision_ms:.0f}",
                str(r.generated_tokens),
                f"{r.generation_ms:.0f}",
                "-" if r.correct is None else ("exact" if r.correct else "WRONG"),
            ]
            for r in results
        ],
    )
    print()

    decision_ms = sum(r.decision_ms for r in results)
    generation_ms = sum(r.generation_ms for r in results)
    total_ms = sum(r.total_ms for r in results)
    tokens = sum(r.generated_tokens for r in results)
    arithmetic = [r for r in results if r.expected is not None]
    routed = sum(1 for r in arithmetic if r.used_tool)

    print()
    print("=" * 78)
    print("MEASURED")
    print("=" * 78)
    print(f"turns                    {len(results)}")
    print(f"decision passes          {len(results)} (one per turn), {decision_ms:.0f} ms total, "
          f"{decision_ms / len(results):.0f} ms each")
    print(f"generation               {tokens} tokens, {generation_ms:.0f} ms total")
    if tokens:
        print(f"                         {generation_ms / tokens:.1f} ms per generated token "
              f"(vs {decision_ms / len(results):.0f} ms per decision)")
    print(f"turn wall clock          {total_ms:.0f} ms, {total_ms / len(results):.0f} ms per turn")
    if total_ms:
        print(f"decision share of it     {decision_ms / total_ms:.1%}")
    if arithmetic:
        right = sum(1 for r in arithmetic if r.correct)
        tool_tokens = sum(r.generated_tokens for r in arithmetic if r.used_tool)
        print(f"arithmetic exactness     {right}/{len(arithmetic)} correct, "
              f"{routed}/{len(arithmetic)} routed to the tool "
              f"({'tool enabled' if assistant.use_tool else 'TOOL DISABLED'})")
        print(f"arithmetic generation    {tool_tokens} tokens on tool-routed turns")
    print()
    print("The decision replaces what an agent normally spends generation on: one pass")
    print("per turn, whatever the label count. The tool path then produces the number")
    print("exactly, in zero generated tokens, instead of approximating it token by")
    print("token -- so on arithmetic the routing decision is both the cheap part and")
    print("the part that removes the errors.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-risk", type=float, default=0.10)
    parser.add_argument("--no-tool", action="store_true", help="let the model do arithmetic itself")
    parser.add_argument(
        "--scorer",
        choices=("local", "hosted"),
        default="local",
        help="hosted reads probabilities from an OpenAI-compatible logprobs endpoint",
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--bench", action="store_true", help="run a scripted set of turns and measure")
    parser.add_argument("--say", default=None)
    args = parser.parse_args()

    if args.report or args.say or args.bench:
        assistant = Assistant(
            model_name=args.model,
            device=args.device,
            target_risk=args.target_risk,
            use_tool=not args.no_tool,
            decision_scorer=args.scorer,
            base_url=args.base_url,
        )
        assistant.report()
        if args.bench:
            return bench(assistant)
        if args.say:
            assistant.show(assistant.turn(args.say))
        return 0

    # Interactive: say what is happening before the slow part, then a compact
    # status instead of the full report. --report prints the whole thing.
    print(f"Loading {args.model} and fitting the calibration...", flush=True)
    started = time.perf_counter()
    assistant = Assistant(
        model_name=args.model,
        device=args.device,
        target_risk=args.target_risk,
        use_tool=not args.no_tool,
        decision_scorer=args.scorer,
        base_url=args.base_url,
    )
    ready_seconds = time.perf_counter() - started

    report = assistant.report_data
    threshold = "unreachable" if report.threshold is None else f"{report.threshold:.3f}"
    print(f"Ready in {ready_seconds:.1f} s.")
    print(
        f"  decision     general or arithmetic, {report.accuracy:.1%} accurate on "
        f"held-out data (ECE {report.calibrated_ece:.4f})"
    )
    print(
        f"  gate         auto-decides {report.coverage:.0%} of turns at "
        f"{report.risk:.1%} error, threshold {threshold}"
    )
    print("  arithmetic   answered by a calculator, not generated")
    print("  everything else is generated prose, one token at a time")
    print("\nType your message, or 'quit' to leave.\n")
    while True:
        try:
            message = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"quit", "exit", "q"}:
            return 0
        if message:
            assistant.show(assistant.turn(message))


if __name__ == "__main__":
    raise SystemExit(main())
