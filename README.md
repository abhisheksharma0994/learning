# learning

Proofs of concept, built and measured: what worked, what did not, and the numbers
for both. Each POC is a self-contained folder you can clone and run, with its
write-up published as a single page on the site.

**Site:** https://abhisheksharma0994.github.io/learning/pocs/

## POCs

| # | POC | What it is | Page |
| --- | --- | --- | --- |
| 01 | [jev-rlcd-poc](pocs/01-jev-rlcd-poc/) | A Jev-shaped RLCD harness: typed decisions, single-pass probabilities, calibrated confidence, risk-coverage evaluation | [RLCD without Jev, measured](https://abhisheksharma0994.github.io/learning/pocs/jev-rlcd-poc/) |

## Running a POC

Each POC runs from its own directory and documents itself. Nothing is shared
between them, so you can clone the whole repo or just read one folder.

```bash
git clone https://github.com/abhisheksharma0994/learning.git
cd learning/pocs/01-jev-rlcd-poc
python examples/demo_report.py          # its README lists every other command
```

01 needs nothing beyond Python 3.10+ for its core layers. Only its optional
local-model backend pulls in torch, together with a model that downloads itself
on first run — see that POC's
[Models section](pocs/01-jev-rlcd-poc/README.md#models-what-gets-downloaded-and-where).

## Layout

```
learning/
  pocs/                   one self-contained folder per POC
    01-jev-rlcd-poc/      code, tests, examples, launcher, its own README
  docs/                   the GitHub Pages site (Jekyll, served from main:/docs)
    _pocs/                one page per POC — the page is the write-up
    _layouts/, assets/    templates and stylesheet
    index.md, pocs.md
```

`pocs/` is the code, `docs/` is the published site: one page per POC at
`/pocs/<slug>/`, and nothing else. The numeric prefix on each POC folder keeps the
order stable as more get added.

## Adding the next POC

Two files, and nothing else to update:

1. **Code** — `pocs/NN-slug/`, self-contained, with its own `README.md` that says
   what to install and how to run it. Assume the reader has only cloned the repo.
2. **The page** — `docs/_pocs/NN-slug.md`, with front matter (`order`, `title`,
   `summary`, `repo_path`, `status`, `date`, `tags`) and the write-up as its body.
   The filename becomes the URL: `docs/_pocs/02-thing.md` → `/pocs/02-thing/`.

The site builds its index and cards from that front matter, so there is no list to
keep in sync. `docs/pocs.md` repeats these steps for reference.

## License

No license file yet, so the default applies: all rights reserved. The code is
published to read, run and learn from — ask first before reusing it in your own
project.
