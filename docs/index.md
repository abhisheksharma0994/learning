---
layout: default
title: null
permalink: /
---

# POCs, measured properly

I build a proof of concept, measure it instead of trusting the pitch, and write
down what actually happened — including the results that argue against the idea I
started with. Each POC ships as runnable code you can clone, with a write-up on
this site.

Every number in these posts comes from a run. Where a claim did not survive
contact with the benchmark, the post says so.

<h2>Proofs of concept</h2>

{% assign pocs = site.pocs | sort: "order" %}
<ul class="cards">
{% for poc in pocs %}
  <li>
    <a class="card" href="{{ poc.url | relative_url }}">
      <span class="card-title">{{ poc.title }}</span>
      <span class="card-summary">{{ poc.summary }}</span>
      <span class="card-meta">{{ poc.status | default: "POC" }}{% if poc.date %} · {{ poc.date | date: "%b %Y" }}{% endif %}</span>
    </a>
  </li>
{% endfor %}
</ul>

{% if site.posts.size > 0 %}
<h2>Latest write-ups</h2>

<ul class="post-list">
{% for post in site.posts limit: 5 %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
  </li>
{% endfor %}
</ul>

<p><a href="{{ '/posts/' | relative_url }}">All write-ups →</a></p>
{% endif %}

<h2>How to run any of them</h2>

Every POC lives under `pocs/` in the [repo]({{ site.github_url }}) as a
self-contained folder, and each one runs from its own directory:

```bash
git clone {{ site.github_url }}.git
cd learning/pocs/01-jev-rlcd-poc        # one folder per POC
python examples/demo_report.py          # each POC README has its own commands
```

Start with the POC's `README.md` — it lists what to install, and what gets
downloaded. The first POC needs nothing beyond Python 3.10+ for its core layers;
only the optional local-model backend pulls in torch, along with a model that
fetches itself from the Hugging Face Hub on first run (953 MB for the 0.5B, 2.9 GB
for the 1.5B) into `$HF_HOME/hub`, or `~/.cache/huggingface/hub` if that is unset.
