---
layout: default
title: Proofs of concept
permalink: /pocs/
---

# Proofs of concept

Each folder under `pocs/` in the [repo]({{ site.github_url }}) is one of these,
self-contained and runnable from its own directory. Each has one page here, and
that page is the full write-up.

{% assign pocs = site.pocs | sort: "order" %}
{% if pocs.size == 0 %}
<p>Nothing published yet.</p>
{% else %}
<ul class="cards">
{% for poc in pocs %}
  <li>
    <a class="card" href="{{ poc.url | relative_url }}">
      <span class="card-title">{{ poc.title }}</span>
      <span class="card-summary">{{ poc.summary }}</span>
      <span class="card-meta">
        <code>{{ poc.repo_path }}</code>{% if poc.status %} · {{ poc.status }}{% endif %}
      </span>
    </a>
  </li>
{% endfor %}
</ul>
{% endif %}

## Adding the next one

Two files, and nothing else to update:

1. **Code** — `pocs/NN-slug/`, self-contained, with its own `README.md` saying what
   to install and how to run it. Zero-padded number so the order stays stable.
2. **The page** — `docs/_pocs/NN-slug.md`, with front matter (`order`, `title`,
   `summary`, `repo_path`, `status`, `date`, `tags`) and the write-up as its body.
   The filename becomes the URL: `docs/_pocs/02-thing.md` → `/pocs/02-thing/`.

The index and the cards are generated from the front matter, so there is no list
to keep in sync. Keep the write-up honest: the failures are the interesting part.
