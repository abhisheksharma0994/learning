# Jev RLCD POC

A harness for **typed, calibrated decisions**: constrain a model to a declared
label set, get a probability for every option out of a single forward pass, then
prove the probabilities are honest — and refuse to answer when they are not.

It is a small, dependency-free take on what Jev advertises — a model that makes
decisions instead of generating strings — built so the claims can be tested
instead of taken on faith. The package inside is `jev_rlcd_poc`.

> **Where this lives.** Part of [learning](https://github.com/abhisheksharma0994/learning),
> in `pocs/01-jev-rlcd-poc/`. Every command below assumes you are in that
> directory:
>
> ```bash
> git clone https://github.com/abhisheksharma0994/learning
> cd learning/pocs/01-jev-rlcd-poc
> ```
>
> The write-up that goes with it: **[RLCD without Jev, measured](https://abhisheksharma0994.github.io/learning/pocs/jev-rlcd-poc/)**
> — what calibrated decisions are good for, and what they are not.

## What it does

Three separable things, and it is worth keeping them separate:

| Layer | Modules | What it gives you |
| --- | --- | --- |
| Decision interface | `scorer`, `backends/` | A closed label set in, a normalized probability per label out. No strings, so no parsing and no type errors. |
| Single-pass scoring | `backends/hf_local` | Every label's logit read from one forward pass. Cost per decision does not grow with label-set size. |
| Calibration + routing | `calibrate`, `metrics`, `decide` | Temperature scaling or histogram binning, ECE/adaptive-ECE/MCE/Brier/NLL, risk-coverage curves, and a threshold that honors a stated error rate. |

The first two are plumbing that any harness can reproduce. The third is where the
actual reliability claim lives, which is why it gets the most code and the most
tests.

## Results

From `python examples/demo_report.py` — a simulated model that is right 72.5% of
the time while stating near-certainty, calibrated on one split and scored on
another:

| variant | acc | ECE | MCE | NLL | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.725 | **0.2696** | 0.7584 | 2.9660 | 0.5408 | 0.0787 |
| temperature (T=4.18) | 0.725 | 0.0625 | 0.1894 | 1.0471 | 0.3799 | 0.0756 |
| histogram (15 bins) | 0.725 | **0.0152** | 0.0327 | 1.0578 | 0.3811 | 0.0790 |

Accuracy is identical across rows because both calibrators are monotone: they
change the numbers attached to decisions, never the decisions. `nll` and `brier`
are proper scoring rules, and they improve in the same direction as ECE, which is
the sanity check that matters.

Two findings from the report that are easy to get wrong:

**A threshold router does not need calibration.** Because monotone calibration
preserves the ranking of decisions, anything that only *orders* decisions —
a confidence threshold, a risk-coverage curve, `ConformalRouter` — reaches the
same operating points either way. In the demo, a target risk of 5% yields 52.7%
coverage on raw probabilities and 54.2% on calibrated ones. If abstention is your
only goal, a harness plus a router is sufficient and calibration is optional.

**Calibration is what makes probabilities composable.** A workflow succeeds only
if every step is right, so its success probability is the product of the
per-step ones — and multiplying confidences is only meaningful if they are
probabilities:

| steps | raw predicted | raw realized | calibrated predicted | calibrated realized |
| --- | --- | --- | --- | --- |
| 1 | 0.995 | 0.725 | 0.714 | 0.725 |
| 5 | 0.975 | 0.199 | 0.192 | 0.199 |
| 10 | 0.950 | 0.031 | 0.037 | 0.031 |
| 20 | 0.902 | 0.000 | 0.001 | 0.000 |

The raw column promises that a 20-step workflow almost always succeeds while it
never succeeded once in hundreds of runs. No amount of harness engineering fixes
that, because the defect is in the numbers.

## Layout

```
src/jev_rlcd_poc/
  scorer.py          LabelSet, Scorer protocol, FunctionScorer, ScoredBatch
  metrics.py         ECE, adaptive ECE, MCE, classwise ECE, NLL, Brier, risk-coverage, AURC
  calibrate.py       TemperatureScaler, HistogramBinner
  calibrated.py      CalibratedDecider: fit, decide, report in three lines
  decide.py          select(), entropy/margin, ConformalRouter
  llm.py             LocalChatModel, for the prose that cannot be templated
  _mathx.py          softmax / logsumexp / 1-D minimizer (stdlib only)
  backends/
    synthetic.py     a skilful-but-overconfident simulated model
    hf_local.py      real local model, single-pass label logits
    openai_logprobs.py  any OpenAI-compatible endpoint, stdlib HTTP only
examples/
  demo_report.py     the full report: raw vs calibrated, invariance, composition
  bring_your_own.py  template: three sections to edit, then a shipping rule
  hf_single_pass.py  the same interface against a real model
  chat.py            chat loop: LLM writes the prose, calibrated decisions route
  assistant.py       one-pass decision, tool for exactness, generated prose
  bench_throughput.py  prefill/decode tok/s with and without the layer, one prompt
  bench_parallel.py  decisions/s vs answers/s as batch size varies
tests/               python -m unittest discover -s tests
```

## Models: what gets downloaded, and where

**You may not need to download anything.** Half of this repo runs on the standard
library alone. Only the five examples that load a real model need weights, and
those fetch them for you on first run.

| What you run | Model | Download |
| --- | --- | --- |
| `demo_report.py`, `bring_your_own.py`, the test suite | none — a simulated model | **0 bytes** |
| `hf_single_pass.py`, `chat.py` | `Qwen/Qwen2.5-0.5B-Instruct` | 953 MB |
| `assistant.py`, `bench_throughput.py`, `bench_parallel.py` | `Qwen/Qwen2.5-1.5B-Instruct` | 2.9 GB |

### It downloads automatically — you do not fetch it by hand

The first time an example runs, `transformers` pulls the weights from the Hugging
Face Hub and caches them. Nothing is gated and no account or token is needed, so
there is no login step:

```bash
.venv/bin/python examples/assistant.py --say "what is 156 * 24"
# first run prints a progress bar while it downloads ~2.9 GB, then answers
# every run after that is offline and instant to start
```

If you would rather see the download before it happens, the `hf` CLI is installed
with the local extra:

```bash
.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct   # 2.9 GB
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct   # 953 MB
```

### Where it lands

By default, in your Hugging Face cache:

| | Path |
| --- | --- |
| Default | `~/.cache/huggingface/hub/` |
| If `HF_HOME` is set | `$HF_HOME/hub/` |
| Example | `HF_HOME=/Users/<name>/Models/huggingface` → `/Users/<name>/Models/huggingface/hub/` |

A model ends up as `models--Qwen--Qwen2.5-1.5B-Instruct/` inside that hub
directory. To keep the weights on an external drive or off your boot disk, point
the cache somewhere else **before** the first run — the models are shared across
projects, so one copy serves them all:

```bash
export HF_HOME=/Volumes/models/huggingface     # tip: put this in your shell profile
```

### Budget the disk and memory

Sizes above are what the two default models actually occupy on disk here. A 1.5B
model in half precision needs roughly its file size in RAM or unified memory once
loaded, plus a little for the KV cache. Bigger models are a straight trade —
better prose and better decisions, slower replies, more memory. The scale runs
≈1 GB (0.5B) → ≈3 GB (1.5B) → ≈6 GB (3B).

### Choosing a different model

Every example takes `--model`, so you can point it at anything on the Hub:

```bash
.venv/bin/python examples/assistant.py --model Qwen/Qwen2.5-3B-Instruct
```

The caveat that matters: labels must tokenize to a **single token**, and the
backend refuses to load rather than silently falling back. That is a property of
the tokenizer, so it can differ between models — expect to rename a label or two
when you swap.

### Download notes worth knowing

- **Faster downloads:** `pip install hf_transfer` then
  `HF_HUB_ENABLE_HF_TRANSFER=1` before the first run.
- **Fully offline after the first run:** set `HF_HUB_OFFLINE=1`. Useful in CI or
  on a plane; it fails loudly instead of hanging if the model is missing.
- **CI and shared machines:** pre-download with `hf download` and set `HF_HOME`
  to a cached directory, so no test run depends on the network.
- **Half-precision loading is the default** on `mps` and `cuda` (float32 on CPU),
  which is where those file sizes come from.

## How to use it

**Look at it first.** No dependencies, no GPU, no API key — the scripts put
`src/` on the path themselves, so any Python 3.10+ works with nothing installed:

```bash
python examples/demo_report.py --print-bins
python -m unittest discover -s tests      # 105 tests, 6 skipped
```

(The six skipped are the local-model tests; run them with
`JEV_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct` set.)

**Then point it at your own decisions.** `examples/bring_your_own.py` is the same
pipeline with three marked sections to edit: your labels, your labeled data, your
scorer. It ends by printing the rule to ship:

```
Rule to ship:
    accept if confidence >= 0.872
    else escalate
  expected: 35.6% handled automatically at 5.0% error
```

**Or wire it in directly.** The shortest path is `CalibratedDecider`, which fits
the calibration and the threshold and reports both:

```python
from jev_rlcd_poc import CalibratedDecider

decider = CalibratedDecider.fit(scorer, cal_states, cal_labels, target_risk=0.05)
print(decider.evaluate(test_states, test_labels))   # held-out numbers

decision, probs, accepted = decider.decide("my invoice looks wrong")
if not accepted:
    escalate_to_human(decision)
```

**Or plug in any scorer.** `scorer.FunctionScorer` adapts a callable, and
`backends.OpenAICompatibleLogprobsScorer` swaps the local model for any hosted
endpoint that returns logprobs — same pipeline, no other change:

```python
from jev_rlcd_poc.backends import OpenAICompatibleLogprobsScorer
from jev_rlcd_poc.scorer import FunctionScorer, LabelSet

LABELS = LabelSet(("billing", "shipping", "returns", "account", "other"))

# Your own callable: mode="probability" (default), or "logits" for raw scores.
my_scorer = FunctionScorer(LABELS, my_classify, mode="logits")

# Or a hosted model, reading its label logprobs directly. Needs OPENAI_API_KEY
# or api_key=. One request per decision, because the answer position is per state.
hosted = OpenAICompatibleLogprobsScorer(
    LABELS, model="gpt-x", system_prompt="Classify the customer message."
)
```

The hosted backend uses only the standard library. Its parsing is covered by
offline tests against canned responses; the live call itself needs a key.

From there the pipeline is four steps — fit, transform, route, report:

```python
from jev_rlcd_poc.calibrate import TemperatureScaler
from jev_rlcd_poc.decide import ConformalRouter
from jev_rlcd_poc.metrics import argmax, ece

print("raw ECE:", ece(raw_test, test_y))
scaler = TemperatureScaler().fit(raw_cal, cal_y)
cal_test = scaler.transform(raw_test)
print("calibrated ECE:", ece(cal_test, test_y))

router = ConformalRouter(target_risk=0.05).fit(
    [max(row) for row in scaler.transform(raw_cal)],
    [argmax(row) == y for row, y in zip(scaler.transform(raw_cal), cal_y)],
)
risk, coverage = router.evaluate(
    [max(row) for row in cal_test],
    [argmax(row) == y for row, y in zip(cal_test, test_y)],
)
print(f"accept if confidence >= {router.threshold_:.3f}: "
      f"{coverage:.1%} of traffic at {risk:.1%} error")
```

**Or use the local single-pass backend** (needs torch and transformers — see
[Models](#models-what-gets-downloaded-and-where) for what else gets downloaded):

```bash
pip install "jev-rlcd-poc[local]"     # into your environment of choice

# Or, from a fresh clone, in the POC directory:
python3 -m venv .venv
.venv/bin/pip install -e '.[local]'

python examples/hf_single_pass.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --labels billing,shipping,returns,account,other
```

Labels must tokenize to a single token. That is deliberate: a multi-token label
would have to be decoded sequentially, which throws away the one property that
makes this fast. The backend refuses to load rather than silently fall back.

### The four things you must get right

1. **Real answers for a few thousand decisions.** Calibration is measured
   against known outcomes. There is no way around this, and a few hundred cases
   is not enough for histogram binning.
2. **Two disjoint splits.** Fit the calibrator on one, report on the other.
   Metrics computed on the split you fitted are fiction.
3. **A closed label set that matches your data.** The stub in
   `bring_your_own.py` deliberately exercises the mismatch check: change `LABELS`
   without changing the data and it tells you why.
4. **Re-measure after any change** to model, prompt, or traffic. A threshold is a
   measurement, not a setting.

## Chat: the model as voice, the harness as gate

`examples/chat.py` answers "can I chat with it?" with the split that makes it
work:

* the **language model** writes the prose — open-ended, unavoidable;
* **one calibrated label set** makes the decision, and the calibrated confidence
  picks the code path.

A single forward pass yields three routes: `chat` → the LLM replies; a team →
auto-routed; below the threshold → escalated to a human rather than guessed at.

```bash
python examples/chat.py --report --diagnostics
python examples/chat.py --say "my invoice has a duplicate charge on it"
python examples/chat.py            # interactive
```

The report with a stock 0.5B model (these numbers are from that run):

| label | n | accuracy | calibrated confidence | auto-accepted |
| --- | --- | --- | --- | --- |
| billing | 29 | 0.97 | 0.588 | 66% |
| shipping | 38 | 0.79 | 0.593 | 79% |
| returns | 36 | 0.08 | 0.399 | 6% |
| account | 35 | 0.83 | 0.446 | 11% |
| other | 44 | 0.00 | 0.432 | 14% |
| chat | 52 | 0.65 | 0.461 | 23% |

**The last two columns are the point.** `billing` is done well and accepted 66%
of the time. `returns` and `other` are beyond this model — and its confidence
there sits below the threshold, so they are accepted 6% and 14% of the time. The
gate suppresses what the model cannot do, which is why 31% of traffic ships at
12% error instead of 47% error at full coverage.

The cost is visible in the same table: `account` is right 83% of the time but
only 11% of it is accepted, because one global temperature cannot repair
per-class miscalibration. Vector or Dirichlet scaling would recover that.

### What a bigger model does and does not fix

Measured on the same held-out set while building this:

| configuration | raw accuracy |
| --- | --- |
| Qwen2.5-0.5B, prompt as-is | 0.567 |
| + label definitions in the system prompt | 0.208 |
| + an explicit answer slot in the user turn | 0.516 |
| Qwen2.5-1.5B, prompt as-is | 0.599 |

Two lessons worth keeping:

1. **Verbose label definitions can hurt.** Describing each label pushed the small
   model to answer with whichever label it had most recently read (0.567 →
   0.208, collapsing onto one class). Terse prompts won.
2. **Label names are a real variable.** Renaming `smalltalk` to `misc` dropped
   that class from 0.78 to 0.03. And `smalltalk` is not a single token at all —
   the backend rejects it rather than falling back to sequential decoding, which
   is how that trap got caught.

The conclusion: reading label logits is a *mechanism*, and it needs a model that
was actually trained to answer in that slot. That is what a purpose-trained
decision model such as Jev is for. With an off-the-shelf chat model your options are a
bigger model, a task with fewer labels, or a better scorer — and the harness is
agnostic about which, because `FunctionScorer` accepts any source of
probabilities.

## Assistant: one-pass decision, generated prose

`examples/assistant.py` is a general chat assistant. The honest version of the
speed claim: generation costs one forward pass per *token*, a decision costs one
pass in total however many labels are in play — so the loop spends a pass on the
decision and generates only the prose.

```bash
python examples/assistant.py --report
python examples/assistant.py --bench             # measured, tool enabled
python examples/assistant.py --bench --no-tool   # same turns, tool disabled
```

Interactive mode prints a compact status instead of the full report, since it is
meant to be launched by double-click:

```
Loading Qwen/Qwen2.5-1.5B-Instruct and fitting the calibration...
Ready in 6.6 s.
  decision     general or arithmetic, 99.5% accurate on held-out data (ECE 0.0069)
  gate         auto-decides 100% of turns at 0.5% error, threshold 0.568
  arithmetic   answered by a calculator, not generated
  everything else is generated prose, one token at a time
```

A double-clickable launcher ships at `Desktop/Assistant.command`. It resolves its
own location, so it works from wherever the repo was cloned — no absolute paths.
On macOS, double-click it in Finder (or symlink it onto your Desktop); from a
terminal, any flag passes through:

```bash
./Desktop/Assistant.command --report
./Desktop/Assistant.command --bench --no-tool
```

It uses the project virtualenv, and if that virtualenv is missing it prints the
three commands to create it and waits rather than closing instantly.

### The design rule this produced

| how the decision is framed | held-out accuracy |
| --- | --- |
| `("answer", "calculate")` — "choose whether this needs a calculator" | 0.670 |
| `("general", "arithmetic")` — "classify the question" | **0.993** |

Asking a model what a text **is** works. Asking it what it should **do** does not.
The action then comes from a lookup on the label in code — which is exactly what a
typed decision is for, and why the label set should describe the input rather
than the response.

### Measured

Twelve scripted turns, Qwen2.5-1.5B-Instruct on MPS:

| | tool enabled | tool disabled |
| --- | --- | --- |
| arithmetic exact | **6/6** | 5/6 |
| generated tokens on arithmetic turns | **0** | 94 |
| generated tokens, all twelve turns | 83 | 175 |
| wall clock, twelve turns | 2154 ms | 4144 ms |
| decision cost | 12 passes, 26 ms each | 12 passes, 25 ms each |

A decision costs about the same as a *single generated token* (26 ms against
22 ms) while replacing the whole planning step. And the arithmetic errors are the
model's, not the harness's: the tool path writes the number with code, so
generation never gets to approximate it.

### Tokens per second, with and without the layer

`examples/bench_throughput.py` runs one long code-generation prompt through the
same model twice — once plain, once through the decision layer — and reports the
prefill and decode rates for both, since "does RLCD make it faster?" deserves a
measurement rather than an assertion:

```bash
python examples/bench_throughput.py --repeats 3
```

Qwen2.5-1.5B-Instruct on MPS, one 357-character prompt asking for a bracket-
balancing function with docstrings and tests, greedy decoding, median of 3:

| config | input tokens | **input tok/s** (prefill) | output tokens | **output tok/s** (decode) | decide ms | total ms |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 112 | **3751** | 400 | **52.5** | — | 7652 |
| with RLCD | 112 **+ 92** | **3751** | 400 | **51.9** | 30 | 7777 |

Rates do not move, and they cannot: prefill throughput is a property of the model
and the prompt length, decode throughput a property of the model and the KV cache.
The decision layer touches neither. What it changes is the token **count**, and on
this prompt the count gets worse — the same 400 output tokens, plus a second
prefill of 92 tokens and one forward pass that buy nothing, because there is no
decision to make. Decode tok/s is derived (output tokens over total minus one
prefill), so read the 52.5-against-51.9 gap as noise, not as a speedup. Generated
text was byte-identical in both configs.

The same script measures the turn type where it does pay, in the same process:

```
prompt                     'what is 827 * 34'  (answer: 28118)

with the decision layer
  decision                 arithmetic p=1.000, 37 ms, one pass
  output tokens            0
  answer                   827 * 34 = 28118.  exact
  total                    38 ms

the same prompt with generation only (no decision, no tool)
  output tokens            21
  answer                   The product of 827 and 34 is 28,558.  WRONG
  total                    466 ms
```

21 tokens and 466 ms replaced by one pass of 37 ms that generates nothing — and
the answer becomes exact, because code placed the number instead of the model
writing it out. That is where the win is: **per decision, not per turn.** A code
prompt is a decision-free turn, and the honest report says so.

### Where "parallel" actually pays

Claims that a decision model gets "higher tokens per second through parallel work"
bundle two different things. `examples/bench_parallel.py` separates them, since a
batch-of-one benchmark cannot see either:

```bash
python examples/bench_parallel.py --batches 1 4 16 64
```

Qwen2.5-1.5B-Instruct on MPS, one question, batch size varied:

| batch | decide ms/req | decisions/s | decode tok/s | answers/s @20 tok | decisions per answer |
| --- | --- | --- | --- | --- | --- |
| 1 | 24.2 | 41.3 | 76.1 | 3.8 | **10.9x** |
| 4 | 7.7 | 130.6 | 152.2 | 7.6 | 17.2x |
| 16 | 5.1 | 197.9 | 574.4 | 28.7 | 6.9x |
| 64 | 4.5 | 222.1 | 1735.6 | 86.8 | **2.6x** |

**Kind one, specific to decision models: parallel over candidates.** One pass reads
every label's logit at once, so the label count is free:

| | ms/decision |
| --- | --- |
| 2 labels | 24.3 |
| 8 labels | 23.8 |

Flat — 4x the labels for 0.98x the time, because it is an `index_select` into a
single forward pass. The sequential alternative, generating one candidate per
label and picking one, cost 647 ms for 23 tokens: **27x** more. A 32-way choice
costs the same as a 2-way choice, where generating 32 candidates costs 32
generations.

**Kind two, not special to decision models: parallel over requests.** Decode
tok/s rises **22.8x** from batch 1 to 64 while decisions/s rises only **5.4x**,
because batching amortises a weight load that decode was already paying *per
token* — at batch 1 decode is memory-bandwidth-bound with idle compute to fill,
while a decision pass is already compute-bound. So under heavy batching
generation closes much of the gap, and "decisions per answer" falls from 11x to
2.6x. Any serving stack gets this from continuous batching; it is not a property
of the model.

What is left, and what actually survives scrutiny: a decision costs **one pass
instead of twenty**, emits **zero tokens you are billed for**, and does not grow a
KV cache. That is a real per-request cost reduction of roughly the answer length
— and it is a claim about the workload's shape, not about tokens per second on
the same weights, which never move.

### What did not work

A second pass judging "is this draft responsive, or does it dodge?" collapsed to
a single class in every framing I tried — 0.63 to 0.79 accuracy at 0.95+
confidence, with one class recalled at 0.00. It is not in the example. A guardrail
that always answers the same thing is worse than no guardrail, and the calibration
report would have shown the collapse (accuracy equal to the majority share,
confidence near 1.0) if I had shipped it.

## What this is not

- **Not RLCD.** No training happens here. This is the post-hoc route to
  calibrated confidence, not the trained-in kind. Expect it to hold up
  in-distribution and to degrade under distribution shift.
- **Not a speed claim.** `hf_local` shows the single-pass mechanism and measures
  it, but the numbers depend entirely on your model and hardware.
- **Not a substitute for semantic validation.** A constrained decoder guarantees
  the shape of an answer, never its correctness.

## Caveats worth internalizing

- A risk target from `ConformalRouter` assumes production traffic is exchangeable
  with the calibration set. Shift voids the guarantee, not just the estimate.
- Temperature scaling is monotone, so it cannot rescue a model whose errors rank
  *above* its correct answers. Check `aurc` before trusting any threshold.
- Adaptive binning keeps confidence ties together. Ignoring that turns an
  arbitrary ordering inside a tie into a calibration error that does not exist —
  the test suite has a regression case for it.
- Small calibration sets make histogram binning noisy; prefer temperature scaling
  below a few thousand labeled decisions.
