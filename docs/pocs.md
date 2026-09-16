---
layout: default
title: Proofs of concept
permalink: /pocs/
---

# Proofs of concept

Each folder under `pocs/` in the [repo]({{ site.github_url }}) is one of these,
self-contained and runnable from its own directory.

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

1. Copy the POC into `pocs/NN-slug/` (zero-padded number, so the order is stable).
2. Add `docs/_pocs/slug.md` with front matter — `order`, `title`, `summary`,
   `repo_path`, `status` — and a short body describing what it does and how to run it.
3. Add the write-up as `docs/_posts/YYYY-MM-DD-slug.md` and point `poc:` at the
   folder it belongs to.

The site builds the index, the POC list and the post list from those files, so
nothing else needs editing.
