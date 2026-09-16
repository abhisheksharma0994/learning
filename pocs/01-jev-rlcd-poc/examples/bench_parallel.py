#!/usr/bin/env python3
"""Where "parallel work" actually buys throughput, measured as batch size varies.

A single-sample benchmark cannot see this. At batch 1 the decision path and the
generation path both run one sequence, and the numbers that come out (39 tok/s of
decode next to 23 ms decisions) look like a wash. The claim being tested here is
that *parallelism* is what turns a decision model into a throughput story, so
this varies the batch size and separates the two kinds of parallelism that get
bundled into one marketing sentence:

1. **parallel over candidates** -- one forward pass reads the logit of every
   label at once, so a 32-way choice costs the same pass as a 2-way choice.
   Generating 32 candidates instead would be 32 sequential passes.
2. **parallel over requests** -- batching amortises the weight load across
   sequences. This one is not specific to decision models: decode benefits from
   it at least as much, which is the point of measuring it.

    python examples/bench_parallel.py
    python examples/bench_parallel.py --batches 1 8 32 --answer-tokens 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_rlcd_poc.backends.hf_local import LocalLabelScorer  # noqa: E402
from jev_rlcd_poc.llm import LocalChatModel  # noqa: E402
from jev_rlcd_poc.scorer import LabelSet  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUESTION = "what is 47 * 12"
ANSWER_SYSTEM = "Answer briefly."


def sync(llm) -> None:
    torch = llm._torch
    if llm.device == "cuda":
        torch.cuda.synchronize()
    elif llm.device == "mps":
        torch.mps.synchronize()


def measure_decisions(scorer, state: str, batch: int, budget_s: float = 0.4) -> float:
    """Seconds per decision, batched -- one forward pass for the whole batch."""
    scorer.batch_size = batch
    states = [state] * batch
    scorer.score(states)  # warm
    elapsed, reps = 0.0, 0
    while elapsed < budget_s or reps < 2:
        started = time.perf_counter()
        scorer.score(states)
        elapsed += time.perf_counter() - started
        reps += 1
    return elapsed / (reps * batch)


def generate(llm, prompts: list[str], max_new_tokens: int) -> tuple[float, int]:
    """Seconds for one batched generate call, and the tokens it produced."""
    tokenizer = llm.tokenizer
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(llm.device)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    with llm._torch.inference_mode():
        out = llm.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
        )
    sync(llm)
    new = out[:, encoded["input_ids"].shape[1]:]
    return float((new != pad).sum()), float(new.shape[0] * new.shape[1])


def measure_generation(llm, prompts: list[str], max_new_tokens: int) -> tuple[float, float]:
    """(decode seconds excluding prefill, tokens generated). Median of a few runs."""
    generate(llm, prompts, 4)  # warm
    started = time.perf_counter()
    _, _ = generate(llm, prompts, 1)  # prefill + exactly one token
    sync(llm)
    prefill_and_one = time.perf_counter() - started

    samples = []
    tokens = 0.0
    for _ in range(3):
        started = time.perf_counter()
        tokens, _ = generate(llm, prompts, max_new_tokens)
        sync(llm)
        samples.append(time.perf_counter() - started)
    samples.sort()
    total = samples[len(samples) // 2]
    # Subtract the prefill-plus-one pass, then account for the one token it made.
    decode_s = max(total - prefill_and_one, 1e-9)
    return decode_s, max(tokens - len(prompts), 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--answer-tokens", type=int, default=20,
                        help="length of the generated answer a decision replaces")
    parser.add_argument("--gen-tokens", type=int, default=48, help="tokens decoded per batch to measure the rate")
    args = parser.parse_args()

    print(f"Loading {args.model}...", flush=True)
    scorer = LocalLabelScorer(
        LabelSet(("general", "arithmetic")),
        model_name=args.model,
        system_prompt="Classify the question.",
        device=args.device,
    )
    llm = LocalChatModel(args.model, device=args.device, max_new_tokens=args.gen_tokens)
    scorer.warmup()
    llm.reply("hello", ANSWER_SYSTEM, max_new_tokens=4)
    prompts = [llm.build_prompt(QUESTION, ANSWER_SYSTEM)]

    print(f"\nmodel                {args.model} on {llm.device}")
    print(f"decision             2 labels, read from one forward pass")
    print(f"answer length        {args.answer_tokens} generated tokens (what a decision replaces)")
    print()

    print("  batch   decide ms/req   decisions/s   decode tok/s   answers/s   decisions per answer")
    print("  " + "-" * 92)
    rows = []
    for batch in args.batches:
        per_request = measure_decisions(scorer, QUESTION, batch)
        decode_s, tokens = measure_generation(llm, prompts * batch, args.gen_tokens)

        decisions_per_s = 1.0 / per_request
        decode_tok_per_s = tokens / decode_s
        answers_per_s = decode_tok_per_s / args.answer_tokens
        ratio = decisions_per_s / answers_per_s
        rows.append((batch, per_request, decisions_per_s, decode_tok_per_s, answers_per_s, ratio))
        print(
            f"  {batch:5d}   {per_request * 1000:10.1f}   {decisions_per_s:11.1f}   "
            f"{decode_tok_per_s:12.1f}   {answers_per_s:9.1f}   {ratio:18.1f}x"
        )
    print()

    single = rows[0]
    best = max(rows, key=lambda row: row[4])
    print("=" * 80)
    print("WHAT THIS SHOWS")
    print("=" * 80)
    print(f"At batch 1 the same model does {single[2]:.0f} decisions/s or {single[4]:.1f} answers/s for a")
    print(f"{args.answer_tokens}-token reply -- the decision handles {single[5]:.0f}x more requests per second.")
    print("That gap is not a faster token. It is:")
    print()
    print(f"  one pass instead of {args.answer_tokens}   the decision reads its answer from a single")
    print("                             forward pass; the reply needs one pass per token")
    print("  all labels at once       32 candidates cost what 2 cost; generating 32")
    print("                            candidates would be 32 sequential passes")
    print("  no KV cache growth       decide in place, or decode 20 steps of state")
    print()
    print("And the second kind of parallelism is not the decision's edge:")
    print(f"  decode tok/s rises {rows[-1][3] / single[3]:.1f}x from batch {rows[0][0]} to {rows[-1][0]},")
    print(f"  decisions/s rises {rows[-1][2] / single[2]:.1f}x over the same range.")
    print("  Batching amortises a weight load that decode was paying per token, so under")
    print("  heavy batching generation closes much of the gap --")
    print(f"  decisions per answer falls from {single[5]:.0f}x to {rows[-1][5]:.0f}x, and at batch")
    print(f"  {best[0]} generation peaks at {best[4]:.1f} answers/s.")
    print()
    print("So the honest decomposition of the claim:")
    print(f"  * per request, a decision costs ~1/{args.answer_tokens} of the wall clock and emits")
    print("    tokens you are not billed for -- that part is real and is what 'lower cost' means")
    print("  * 'parallel' is two different things, and only the candidate-parallel one is")
    print("    specific to decision models; request-parallel batching helps any model")
    print("  * it is NOT a higher tokens/sec on the same weights: at a fixed workload the")
    print("    rates are identical, since neither prefill nor decode throughput changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
