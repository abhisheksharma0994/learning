# learning

Proofs of concept, built and measured: what worked, what did not, and the numbers
for both. Each POC is a self-contained folder you can clone and run, with a
write-up published on the site.

**Site:** https://abhisheksharma0994.github.io/learning/

## POCs

| # | POC | What it is | Write-up |
| --- | --- | --- | --- |
| 01 | [jev-rlcd-poc](pocs/01-jev-rlcd-poc/) | A Jev-shaped RLCD harness: typed decisions, single-pass probabilities, calibrated confidence, risk-coverage evaluation | [RLCD without Jev, measured](https://abhisheksharma0994.github.io/learning/posts/jev-rlcd-measured/) |

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
    _posts/               write-ups, one per POC
    _pocs/                one page per POC; drives the site indexes
    _layouts/, assets/    templates and stylesheet
    index.md, pocs.md, posts.md
```

`pocs/` is the code, `docs/` is the published site. The numeric prefix on each
POC folder keeps the order stable as more get added.

## Adding the next POC

1. **Code** — `pocs/NN-slug/`, self-contained, with its own `README.md` that says
   what to install and how to run it. Assume the reader has only cloned the repo.
2. **Site page** — `docs/_pocs/slug.md` with front matter (`order`, `title`,
   `summary`, `repo_path`, `status`, optionally `post`) and a short body.
3. **Write-up** — `docs/_posts/YYYY-MM-DD-slug.md` with front matter (`title`,
   `date`, `description`, `poc`, `tags`).

Both indexes and the post list are generated from those files, so there is
nothing else to update. `docs/pocs.md` repeats this list for reference.

## License

No license file yet, so the default applies: all rights reserved. The code is
published to read, run and learn from — ask first before reusing it in your own
project.
