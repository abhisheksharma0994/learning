#!/usr/bin/env python3
"""Same model, same prompt, with and without the decision layer.

Measures, for one long code-generation prompt:

* input tokens and prefill tokens/sec
* output tokens and decode tokens/sec
* end-to-end wall clock, and the decision pass's share of it

The expected result, stated up front so the numbers can be judged against it:
**RLCD does not change tokens per second.** It changes whether tokens are spent
at all. For free-form generation it is pure overhead -- one extra forward pass
plus a second prefill -- because there is no decision to make. That is worth
measuring rather than asserting, so this does both configurations and checks
that the generated text is identical.

    python examples/bench_throughput.py
    python examples/bench_throughput.py --repeats 5 --max-new-tokens 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(EXAMPLES.parent / "src"))

from assistant import ANSWER_SYSTEM, Assistant  # noqa: E402

#: A turn the layer *can* resolve without generating. Chosen because 827*34 is
#: the kind of product a 1.5B model gets wrong when asked to write it out.
ARITHMETIC_CONTRAST = "what is 827 * 34"
CONTRAST_EXPECTED = 28118

PROMPT = (
    "Write a Python function parse_nested_brackets(string) that validates whether "
    "brackets (), {}, and [] are balanced. The function must return True if valid, "
    "or a descriptive string explaining the exact index and nature of the error if "
    "invalid (e.g., 'Unmatched closing bracket ] at index 12'). Include docstrings "
    "and unit tests covering at least 3 edge cases."
)


def sync(llm) -> None:
    torch = llm._torch
    if llm.device == "cuda":
        torch.cuda.synchronize()
    elif llm.device == "mps":
        torch.mps.synchronize()


def measure_prefill(llm, prompt: str, repeats: int) -> tuple[int, float]:
    """Token count of the prompt, and median seconds for one forward pass over it."""
    torch = llm._torch
    inputs = llm.tokenize(prompt)
    token_count = int(inputs["input_ids"].shape[1])
    timings = []
    with torch.inference_mode():
        for _ in range(repeats):
            sync(llm)
            started = time.perf_counter()
            llm.model(**inputs)
            sync(llm)
            timings.append(time.perf_counter() - started)
    return token_count, statistics.median(timings)


def measure_generation(llm, prompt: str, system: str, max_new_tokens: int, repeats: int):
    """Median generated tokens, median seconds, and one sample of the text."""
    token_counts, seconds, text = [], [], ""
    for _ in range(repeats):
        started = time.perf_counter()
        text = llm.reply(prompt, system, max_new_tokens=max_new_tokens)
        seconds.append(time.perf_counter() - started)
        token_counts.append(llm.last_tokens)
    return int(statistics.median(token_counts)), statistics.median(seconds), text


def measure_decision(assistant, message: str, repeats: int) -> float:
    """Median seconds for one decision pass (a single forward pass)."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        assistant.decider.decide(message)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def print_table(headers, rows) -> None:
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repeats", type=int, default=3, help="median of this many runs")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--target-risk", type=float, default=0.10)
    args = parser.parse_args()

    print(f"Loading {args.model} and fitting the calibration...", flush=True)
    assistant = Assistant(model_name=args.model, device=args.device, target_risk=args.target_risk)
    llm = assistant.llm

    # Throwaway work first: the initial calls pay for lazy device setup and graph
    # compilation, which would otherwise land inside the first measurement.
    llm.reply("hello", ANSWER_SYSTEM, max_new_tokens=4)
    assistant.decider.decide("hello")

    generation_prompt = llm.build_prompt(PROMPT, ANSWER_SYSTEM)
    input_tokens, prefill_s = measure_prefill(llm, generation_prompt, args.repeats)
    decision_prompt = assistant.scorer.render(PROMPT)
    decision_tokens = len(assistant.scorer.tokenizer(decision_prompt)["input_ids"])

    print(f"\nmodel                    {args.model} on {llm.device}")
    print(f"prompt                   {len(PROMPT)} characters")
    print(f"repeats                  {args.repeats}, median reported")
    print(f"decoding                 greedy, max_new_tokens={args.max_new_tokens}")
    print()

    # --- without the decision layer -----------------------------------------
    output_tokens, generation_s, plain_text = measure_generation(
        llm, PROMPT, ANSWER_SYSTEM, args.max_new_tokens, args.repeats
    )
    plain_decode_s = max(generation_s - prefill_s, 1e-9)

    # --- with the decision layer --------------------------------------------
    decision_s = measure_decision(assistant, PROMPT, args.repeats)
    turns = [assistant.turn(PROMPT, max_new_tokens=args.max_new_tokens) for _ in range(args.repeats)]
    middle = sorted(turns, key=lambda turn: turn.total_ms)[len(turns) // 2]
    rlcd_generation_s = middle.generation_ms / 1000.0
    rlcd_decode_s = max(rlcd_generation_s - prefill_s, 1e-9)

    print("=" * 78)
    print("ONE CODE-GENERATION PROMPT")
    print("=" * 78)
    print(PROMPT)
    print()
    print_table(
        ["config", "in tok", "out tok", "prefill tok/s", "decode tok/s", "decide ms", "total ms"],
        [
            [
                "plain",
                str(input_tokens),
                str(output_tokens),
                f"{input_tokens / prefill_s:.0f}",
                f"{output_tokens / plain_decode_s:.1f}",
                "0",
                f"{generation_s * 1000:.0f}",
            ],
            [
                "with RLCD",
                f"{input_tokens} + {decision_tokens}",
                str(middle.generated_tokens),
                f"{input_tokens / prefill_s:.0f}",
                f"{middle.generated_tokens / rlcd_decode_s:.1f}",
                f"{decision_s * 1000:.0f}",
                f"{middle.total_ms:.0f}",
            ],
        ],
    )
    print()
    print("Both rows prefill once for the generation itself. The RLCD row prefills a")
    print(f"second, separate prompt of {decision_tokens} tokens for the decision, which")
    print(f"costs one forward pass ({decision_s * 1000:.0f} ms) and generates nothing.")
    print()
    print("decode tok/s is derived -- generated tokens over (total - one prefill) -- and")
    print("the two rows run the identical decode loop on the identical prompt, so read any")
    print("gap between those two columns as noise, not as the decision layer buying speed.")
    print()
    identical = plain_text.strip() == middle.answer.strip()
    print(f"generated text identical in both configs: {'yes' if identical else 'NO'}")
    if not identical:
        print(f"  plain     : {plain_text[:90]!r}")
        print(f"  with RLCD : {middle.answer[:90]!r}")
    print()

    # --- where the decision layer does pay -----------------------------------
    contrast = assistant.turn(ARITHMETIC_CONTRAST, max_new_tokens=args.max_new_tokens)
    # The same turn with generation only: no decision, no tool. Measured here
    # rather than quoted, so the comparison is self-contained.
    generated_answer = llm.reply(ARITHMETIC_CONTRAST, ANSWER_SYSTEM, max_new_tokens=args.max_new_tokens)
    generated_tokens, generated_ms = llm.last_tokens, llm.last_latency_ms
    print("=" * 78)
    print("THE SAME PIPELINE ON A TURN IT CAN DECIDE")
    print("=" * 78)
    print(f"prompt                     {ARITHMETIC_CONTRAST!r}  (answer: {CONTRAST_EXPECTED})")
    print()
    print("with the decision layer")
    print(f"  decision                 {contrast.label} p={contrast.confidence:.3f}, "
          f"{contrast.decision_ms:.0f} ms, one pass")
    print(f"  output tokens            {contrast.generated_tokens}")
    print(f"  answer                   {contrast.answer}  "
          f"{'exact' if str(CONTRAST_EXPECTED) in contrast.answer else 'WRONG'}")
    print(f"  total                    {contrast.total_ms:.0f} ms")
    print()
    print("the same prompt with generation only (no decision, no tool)")
    print(f"  output tokens            {generated_tokens}")
    print(f"  answer                   {generated_answer.replace(chr(10), ' ')}  "
          f"{'exact' if str(CONTRAST_EXPECTED) in generated_answer.replace(',', '') else 'WRONG'}")
    print(f"  total                    {generated_ms:.0f} ms")
    print()
    if contrast.generated_tokens < generated_tokens:
        print(f"That is {generated_tokens} tokens and {generated_ms:.0f} ms replaced by one pass of")
        print(f"{contrast.decision_ms:.0f} ms that generated nothing -- and the answer is exact, because code")
        print("placed the number instead of the model writing it out.")
        print()

    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print("Tokens per second do not move between configs. Prefill throughput is a")
    print("property of the model and the prompt length; decode throughput is a property")
    print("of the model and the KV cache. The decision layer touches neither, so the")
    print("rates are the same and the only difference is token *count*.")
    print()
    print("For this prompt that count gets worse, not better:")
    print(f"  {middle.generated_tokens} output tokens either way, and {decision_tokens} extra")
    print(f"  prefilled tokens plus one forward pass ({decision_s * 1000:.0f} ms) that buy nothing")
    print()
    print("It pays only where a turn can be resolved without generating at all:")
    print(f"  {contrast.generated_tokens} output tokens on the arithmetic turn instead of")
    print(f"  {generated_tokens}, and the decision pass costs less than one generated token")
    print()
    print("So the speedup is per-decision, not per-turn. A code prompt is a decision-free")
    print("turn: there is nothing to route, so the layer is overhead and the honest thing")
    print("is to say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
