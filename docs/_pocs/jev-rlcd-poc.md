---
order: 1
title: "Jev RLCD POC: a calibrated decision harness"
summary: >-
  A Jev-shaped decision harness, built without Jev. Constrain a model to a
  declared label set, read every label's probability out of one forward pass,
  calibrate it on labelled data, and refuse to answer when the confidence is low.
repo_path: pocs/01-jev-rlcd-poc
status: complete
date: 2026-09-15
post: /posts/jev-rlcd-measured/
tags: [jev, rlcd, calibration, llm, evals]
---

## What it is

The package inside is `jev_rlcd_poc`: a small, mostly dependency-free take on what
Jev advertises — a model that makes decisions instead of generating strings — built
so the claims can be tested rather than taken on faith.

Three separable layers, and keeping them apart is most of the point:

| Layer | What it gives you |
| --- | --- |
| Decision interface | A closed label set in, a probability per label out. No strings, so no parsing and no type errors. |
| Single-pass scoring | Every label's logit read from one forward pass, so cost per decision does not grow with the label count. |
| Calibration + routing | Temperature scaling or histogram binning, ECE/Brier/NLL, risk-coverage curves, and a threshold that honours a stated error rate. |

The first two are plumbing any harness can reproduce. The third is where the
reliability claim actually lives, which is why it gets most of the code and tests.

## What it showed

- **Calibration changes the numbers, never the decisions.** Held-out ECE dropped
  from 0.2696 to 0.0152 with accuracy identical at 0.725, because both calibrators
  are monotone.
- **A threshold router does not need calibration.** Ranking is preserved, so
  abstention works either way — 52.7% against 54.2% coverage at a 5% target. The
  payoff of calibration is that probabilities become *composable*: raw confidence
  promised a 20-step workflow would almost always succeed, and it never once did.
- **The gate suppresses what the model cannot do.** On six intents with a 0.5B
  model, the two classes it was dead on were auto-accepted only 6% and 14% of the
  time, so 31% of traffic shipped at 12% error instead of 47% at full coverage.
- **Framing decides whether it works at all.** "Classify the question" scored
  0.993 where "choose whether this needs a calculator" scored 0.670. Ask what the
  input *is*, never what the model should *do*.
- **Tokens per second never move.** The same 400-token patch took 7652 ms plain
  and 7777 ms through the decision layer: a decision replaces a *turn*, not a
  generation.
- **Some of it does not work.** A second pass judging whether a draft answers the
  question collapsed to a single class in every framing, so it was left out rather
  than shipped.

## How to run it

```bash
git clone https://github.com/abhisheksharma0994/learning.git
cd learning/pocs/01-jev-rlcd-poc

# Nothing to install and nothing to download — stock Python 3.10+
python examples/demo_report.py --print-bins
python -m unittest discover -s tests        # 105 tests, 6 skipped

# The model-backed examples need torch + transformers
python3 -m venv .venv && .venv/bin/pip install -e '.[local]'
.venv/bin/python examples/chat.py --report --diagnostics    # 0.5B, 953 MB
.venv/bin/python examples/assistant.py --bench              # 1.5B, 2.9 GB
```

The weights download themselves from the Hugging Face Hub on first run, into
`$HF_HOME/hub` or `~/.cache/huggingface/hub`. Nothing is gated and no token is
needed. The POC's README has the pre-download command, how to move the cache, and
how to pick a different model.

## Limits worth knowing

This is the post-hoc route to calibrated confidence, not RLCD itself: nothing is
trained, so expect it to hold in-distribution and to degrade under distribution
shift. It also needs labelled outcomes to measure against — a few thousand
decisions, split into two disjoint halves — and that is the one requirement you
cannot engineer around.
