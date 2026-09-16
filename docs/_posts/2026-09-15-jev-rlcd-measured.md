---
title: "RLCD without Jev, measured: what calibrated decisions are good for (and what they are not)"
date: 2026-09-15
description: >-
  TypeSafe's Jev claims typed decisions with honest confidence. I built the
  Jev-shaped harness without Jev, measured every claim with a real model, and kept
  the results that argue against the easy version - including where the decision
  layer turns out to be pure overhead.
poc: pocs/01-jev-rlcd-poc
poc_name: "Jev RLCD POC"
tags: [jev, rlcd, calibration, llm, evals]
---

TypeSafe AI recently announced Jev, a "System One" model for **RLCD** —
reinforcement learning for calibrated decisions — with the kind of claims that are
hard to argue with and hard to check: typed decisions, no type errors *by
construction*, 70–500 ms decisions, and higher throughput at lower cost through
parallel work.

I wanted to know which of those claims survive contact with an off-the-shelf
model and a small harness, so I built one and measured everything. This post is
the result: what RLCD is, what a harness reproduces, what it provably cannot,
how to chat with the thing, and where it genuinely earns its place in a system.

Everything below is measured on Qwen2.5-1.5B-Instruct (and 0.5B where noted) on
an M1 Mac with MPS. The code is in this repo and you can run all of it without a
GPU, an API key, or an internet connection.

---

## First, the honest definition

**RLCD is a training method.** The bet is that a model can often do a task 95% of
the time but cannot tell you *when it is in the 5%* — and if it cannot say that,
you cannot automate the task, because you have no basis for routing the uncertain
cases to a human. So you train the honesty in, using a proper scoring rule as the
reward signal.

**What this repo is not:** it is not RLCD. Nothing here is trained. Every number
comes from the complementary, unglamorous route:

1. constrain the model to a **closed label set** so it cannot emit a malformed
   answer;
2. read **all label probabilities out of one forward pass**;
3. **calibrate** those probabilities post-hoc on labeled data;
4. **route** on the calibrated confidence, and abstain when it is low.

That gets you the interface of a Jev-shaped decision model and an honest
measurement of your own reliability. It does not get you calibration trained *into* the weights,
which is the part that holds up under distribution shift.

Keeping those two apart is the whole point of the exercise. Here is the claim
decomposed into what is reproducible, and how much of it is real:

| claim | reproducible with a harness? | verdict |
| --- | --- | --- |
| Typed output, no type errors | **Yes, exactly** | Grammar-constrained decoding. Free. The least novel part. |
| One pass scores every candidate | **Yes, exactly** | Measured flat: 8 labels cost 0.98x of 2 labels. |
| Fast and cheap per decision | **Partly** | True per decision, false per turn — see [below](#the-claim-that-does-not-survive). |
| Probabilities you can act on | **No** | Raw logprobs are overconfident. This needs calibration, and post-hoc calibration is not RLCD. |

---

## The three layers, kept separate

The most useful thing I learned building this is that the stack has three
separable layers, and confusing them is how the claims get inflated:

| Layer | What it gives you | Can a harness reproduce it? |
| --- | --- | --- |
| Decision interface (closed label set → probability per label) | No strings, so no parsing, no type errors | Yes — that is just constrained decoding |
| Single-pass scoring (every label's logit from one forward pass) | Cost per decision independent of label-set size | Yes — measurable, and I measured it |
| Calibration + routing (temperature scaling, ECE, risk-coverage) | A confidence you can set a threshold on | Only post-hoc; the trained-in version is RLCD |

The first two are plumbing. The third is where the actual reliability claim
lives, which is why it is where most of this repo's code and tests went.

---

## Run it first, with nothing installed

No dependencies for any of this — the scripts put `src/` on the path themselves,
so any Python 3.10+ works with nothing installed:

```bash
python examples/demo_report.py --print-bins
python -m unittest discover -s tests      # 105 tests, 6 skipped
```

`demo_report.py` makes the case for calibration better than prose can. A simulated
model that is right 72.5% of the time while stating near-certainty:

| variant | accuracy | ECE | MCE | NLL | Brier |
| --- | --- | --- | --- | --- | --- |
| raw | 0.725 | **0.2696** | 0.7584 | 2.9660 | 0.5408 |
| temperature (T=4.18) | 0.725 | 0.0625 | 0.1894 | 1.0471 | 0.3799 |
| histogram (15 bins) | 0.725 | **0.0152** | 0.0327 | 1.0578 | 0.3811 |

Accuracy is identical across the rows, because both calibrators are monotone:
they change the numbers attached to decisions, never the decisions. ECE drops
from 0.27 to 0.015 with the decisions unchanged.

The clearest way to see why that matters is composition. A workflow succeeds only
if *every* step succeeds, so its success probability is the product of the
per-step ones — and multiplying confidences is only meaningful if they are
probabilities:

| steps | raw predicted | raw realized | calibrated predicted | calibrated realized |
| --- | --- | --- | --- | --- |
| 1 | 0.995 | 0.725 | 0.714 | 0.725 |
| 10 | 0.950 | 0.031 | 0.037 | 0.031 |
| 20 | 0.902 | 0.000 | 0.001 | 0.000 |

Raw confidence promises that a 20-step workflow almost always works. It did not
succeed once in hundreds of runs. No amount of prompt or harness engineering
fixes that, because the defect is in the numbers, not the plumbing.

---

## Chat with it

You can chat with a calibrated decision layer. The trick is to split the roles
instead of merging them:

- the **language model** writes the prose — open-ended and unavoidable;
- **one calibrated label set** makes the decision, and its calibrated confidence
  picks the code path.

`examples/chat.py` scores your message once and routes it three ways: a team owns
it → auto-routed; it is `chat` → the LLM writes a reply; below threshold →
escalated to a human rather than guessed at.

```bash
python examples/chat.py --report --diagnostics
python examples/chat.py --say "the courier said they delivered but nothing arrived"
python examples/chat.py            # interactive
```

### A real transcript

```
one decision: which team owns this message, if any
  labelled decisions          234 calibration / 234 held out
  accuracy on held-out data   0.530
  ECE raw -> calibrated       0.2319 -> 0.1221
  accept if confidence >=     0.525
  held-out coverage / risk    31.2% auto-routed at 12.3% error

you  > the courier said they delivered but nothing arrived
     [decision] shipping p=0.815 (accept)
     [top 3]    shipping 0.815, billing 0.096, returns 0.053
bot  > Passing this to the shipping team (confidence 82%).
```

That message was invented on the spot, not drawn from the calibration set — the
score is 0.815, above the fitted 0.525 threshold, so the route is taken and the
bot says so with the number attached.

### The assistant, with a tool behind the decision

`examples/assistant.py` adds the part that makes the speed claim real: the decision
routes arithmetic to an actual evaluator, so the number is placed by code and
never generated.

```
you  > what is 156 * 24
     [decide] arithmetic p=1.000 (24 ms, one pass, tool used)
bot  > 156 * 24 = 3744.

you  > why does the moon affect tides
     [decide] general p=1.000 (24 ms, one pass)
     [answer] 31 tokens in 671 ms (generated)
bot  > The Moon's gravity pulls on Earth's oceans, causing them to bulge outward
       and creating high tides. This is known as the tidal effect.
```

The 1.5B weights are fetched from the Hugging Face Hub on first run: 2.9 GB into
`$HF_HOME/hub`, or `~/.cache/huggingface/hub` if that is unset, cached once and
reused by every later run. `chat.py`'s 0.5B model is 953 MB. Nothing is gated and
no token is needed, and the POC's README has the pre-download command, the
cache-location options, and how to pick a different model with `--model`.

There is a double-clickable launcher at `Desktop/Assistant.command` — it resolves
its own location, so it works from wherever you clone the repo, and it warns you
before the first-run download — that starts this in interactive mode against the
1.5B model. It loads and fits the calibration
in about 7 seconds, then greets you with what it measured:

```
Ready in 6.8 s.
  decision     general or arithmetic, 99.5% accurate on held-out data (ECE 0.0069)
  gate         auto-decides 100% of turns at 0.5% error, threshold 0.568
  arithmetic   answered by a calculator, not generated
  everything else is generated prose, one token at a time
```

---

## Where RLCD-shaped decisions are genuinely good

The pattern that works is **turn a question into a closed set you declare in
advance**, then let the calibrated confidence decide whether to act on it.

| use case | why it fits |
| --- | --- |
| **Routing a ticket to a team** | 5–6 labels, high volume, and a wrong route is recoverable — the confidence tells you which ones to check. |
| **Deciding whether to escalate** | This is the killer app. You do not need 99% accuracy; you need to know *which* 30% is safe to automate. |
| **Tool or model selection** | One pass replaces a planning chain, then the action is a dict lookup in your code. |
| **Computing instead of generating** | `what is 827 * 34`: 21 tokens, 466 ms, and the model said **28,558**. The decision path: one 37 ms pass, zero output tokens, exact. |
| **Constrained classification** | Intent, category, sentiment, jailbreak/no-jailbreak, which of 20 failure modes explains this log line. |
| **Anywhere output tokens are the cost** | A decision emits ~1 token instead of ~20, and does not grow a KV cache. |

### Where it is the wrong tool

| anti-pattern | what happens |
| --- | --- |
| **Generating long free-form output** | Measured: a 400-token code patch took 7652 ms plain and 7777 ms with the decision layer — same 400 tokens, plus 92 prefilled and one forward pass that buy nothing. A code prompt is a decision-free turn. |
| **Judging quality** ("is this patch good?", "is this draft responsive?") | I tried it. It collapsed to one class in every framing: 0.63–0.79 accuracy at 0.95+ confidence, with one class recalled at 0.00. I left it out rather than ship a bluff. |
| **Open-ended planning** | There is no closed label set to read logits from. |
| **Reasoning that needs intermediate tokens** | That is exactly the capability the design gives up. |

---

## The claim that does not survive

"Higher tokens per second through parallel work" needs splitting into two
different things, and only one of them is special to decision models.

**Parallel over candidates — real, and genuinely theirs.** One pass reads every
label's logit at once, so the label count is free:

| | ms/decision |
| --- | --- |
| 2 labels | 24.3 |
| 8 labels | 23.8 |

Flat: 4x the labels for **0.98x** the time, because it is an `index_select` into a
single forward pass. The sequential alternative — generate one candidate per label
and pick one — cost **647 ms for 23 tokens, 27x more**. A 32-way choice costs what
a 2-way choice costs, where generating 32 candidates costs 32 generations.

**Parallel over requests — real, but not theirs.** Batch size varied:

| batch | decide ms/req | decisions/s | decode tok/s | answers/s @20 tok | decisions per answer |
| --- | --- | --- | --- | --- | --- |
| 1 | 24.2 | 41.3 | 76.1 | 3.8 | **10.9x** |
| 16 | 5.1 | 197.9 | 574.4 | 28.7 | 6.9x |
| 64 | 4.5 | 222.1 | 1735.6 | 86.8 | **2.6x** |

Decode tok/s rises **22.8x** with batching while decisions/s rises **5.4x**,
because batching amortises a weight load that decode was paying *per token*.
Under heavy batching generation closes most of the gap, and any serving stack
gets that from continuous batching. It is not a property of the model.

**And the rates on the same weights never move.** Same model, same prompt, same
400 output tokens, with and without the decision layer:

| config | input tokens | output tokens | prefill tok/s | decode tok/s | decide ms | total ms |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 112 | 400 | 3751 | 52.5 | — | 7652 |
| with RLCD | 112 **+ 92** | 400 | 3751 | 51.9 | 30 | 7777 |

Generated text was byte-identical. Prefill throughput is a property of the model
and the prompt length; decode throughput a property of the model and the KV cache.
The decision layer touches neither.

**What survives:** a decision costs **one pass instead of twenty**, emits **zero
tokens you are billed for**, and does not grow a KV cache. That is a real
per-request cost reduction of roughly the answer length — a claim about the shape
of your workload, not about tokens per second. The speedup is **per decision, not
per turn**, and any honest benchmark should say so out loud.

---

## Three findings that changed the design

**1. Ask what the input *is*, never what the model should *do*.**

| framing | held-out accuracy |
| --- | --- |
| `("answer", "calculate")` — "choose whether this needs a calculator" | 0.670 |
| `("general", "arithmetic")` — "classify the question" | **0.993** |

The action then comes from a lookup on the label in your code. That is what a
typed decision is *for*, and why your labels should describe the input rather than
the response.

**2. A threshold router does not need calibration.** Because monotone calibration
preserves ranking, a confidence threshold, a risk-coverage curve and a conformal
router reach the same operating points either way — 52.7% coverage versus 54.2%
at a 5% target. If abstention is all you want, a harness plus a router is enough
and you can skip the calibrator. Calibration is what makes probabilities
**composable**, which is where it pays.

**3. The gate suppresses exactly what the model cannot do.** With a stock 0.5B
model on six labels:

| label | n | accuracy | calibrated confidence | auto-accepted |
| --- | --- | --- | --- | --- |
| billing | 29 | 0.97 | 0.588 | **66%** |
| shipping | 38 | 0.79 | 0.593 | **79%** |
| returns | 36 | 0.08 | 0.399 | **6%** |
| account | 35 | 0.83 | 0.446 | 11% |
| other | 44 | 0.00 | 0.432 | **14%** |
| chat | 52 | 0.65 | 0.461 | 23% |

The model is dead on `returns` and `other`, and its confidence there sits below
the threshold — so 6% and 14% get accepted. Net: **31% of traffic ships at 12%
error** instead of 47% error at full coverage. The gate knows what it cannot do.
The cost is visible too: `account` is right 83% of the time but only 11% is
accepted, because one global temperature cannot repair per-class miscalibration.

**And some things simply do not work at this scale.** Terse prompts beat verbose
ones: adding label definitions to the system prompt pushed accuracy from 0.567 to
**0.208**, because the small model just answered with whichever label it had most
recently read. Label *names* are a real variable — renaming `smalltalk` to `misc`
dropped that class from 0.78 to 0.03. And `smalltalk` is not a single token at
all; the backend refuses to load rather than silently falling back to sequential
decoding, which is how that trap got caught.

The lesson: reading label logits is a *mechanism*, and it needs a model trained to
answer in that slot. That is what a purpose-trained decision model such as Jev is
for, and why a stock 0.5B chat model cannot be talked into the role.

---

## The four things you must get right

1. **Real answers for a few thousand decisions.** Calibration is measured against
   known outcomes. There is no way around this, and a few hundred cases is not
   enough for histogram binning.
2. **Two disjoint splits.** Fit the calibrator on one, report on the other.
   Metrics computed on the split you fitted are fiction.
3. **A closed label set matching your data, with single-token labels.** The
   backend errors out rather than silently degrading.
4. **Re-measure after any change** to model, prompt, or traffic. A threshold is a
   measurement, not a setting.

The shortest path to your own data is `examples/bring_your_own.py` — three marked
sections to edit, and it ends by printing the rule to ship:

```
Rule to ship:
    accept if confidence >= 0.872
    else escalate
  expected: 35.6% handled automatically at 5.0% error
```

Or wire it in directly:

```python
from jev_rlcd_poc import CalibratedDecider

decider = CalibratedDecider.fit(scorer, cal_states, cal_labels, target_risk=0.05)
print(decider.evaluate(test_states, test_labels))   # held-out numbers

decision, probs, accepted = decider.decide("my invoice looks wrong")
if not accepted:
    escalate_to_human(decision)                     # do not ship a guess
```

Nothing above is tied to the local model. `FunctionScorer` adapts any callable,
and `OpenAICompatibleLogprobsScorer` reads label logprobs from any OpenAI-
compatible endpoint using only the standard library — same pipeline, same
calibration, same routing, no other change. That is the payoff of the harness
being scorer-agnostic.

---

## What this is not

- **Not RLCD.** No training. This is the post-hoc route to calibrated confidence.
  Expect it to hold in-distribution and to degrade under distribution shift.
- **Not a speed claim for your agent.** If your agent's time is dominated by
  writing code, no decision layer moves that number. It is a router and a gate.
- **Not a substitute for validation.** A constrained decoder guarantees the shape
  of an answer, never its correctness.

And the caveats worth internalizing: a risk target assumes production traffic is
exchangeable with your calibration set (shift voids the guarantee, not just the
estimate); temperature scaling is monotone, so it cannot rescue a model whose
errors rank *above* its correct answers — check `aurc` before trusting a
threshold; and small calibration sets make histogram binning noisy, so prefer
temperature scaling below a few thousand labeled decisions.

---

## Start here

Run these from `pocs/01-jev-rlcd-poc/` inside the repo — the POC's own
[README](https://github.com/abhisheksharma0994/learning/tree/main/pocs/01-jev-rlcd-poc)
has the clone command, and its
[Models section](https://github.com/abhisheksharma0994/learning/tree/main/pocs/01-jev-rlcd-poc#models-what-gets-downloaded-and-where)
covers what the model-backed examples download (953 MB for the 0.5B, 2.9 GB for
the 1.5B) and where it lands. The first two commands below need no download at all.

```bash
# See calibration matter, with no dependencies at all
python examples/demo_report.py --print-bins

# Chat, with the decision layer visible on every turn
python examples/chat.py --report --diagnostics
python examples/chat.py --say "the courier said they delivered but nothing arrived"

# The assistant, with a tool behind the decision, and the benchmarks
python examples/assistant.py --bench
python examples/bench_throughput.py --repeats 3
python examples/bench_parallel.py --batches 1 4 16 64
```

The first two commands need nothing installed beyond Python 3.10+. The rest load
a real model, so they need `pip install "jev-rlcd-poc[local]"` (torch and
transformers) — or swap the scorer for the hosted backend and skip the local
weights entirely.

**The one-sentence version:** a calibrated decision layer is not a faster model —
it is a smart if-statement with an honest confidence attached, and the honest
confidence is the part that lets you automate anything.

Everything in this post is reproducible from this folder, and every number came
from a run — including the ones that argue against the easy version of the story.
