---
layout: default
title: Write-ups
---

# Write-ups

One post per POC, plus anything else worth writing down. Long form, with the
benchmarks and the negative results included.

{% if site.posts.size == 0 %}
<p>Nothing published yet.</p>
{% else %}
<ul class="post-list">
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
    {% if post.tags.size > 0 %}<span class="tags">{{ post.tags | join: " · " }}</span>{% endif %}
  </li>
{% endfor %}
</ul>
{% endif %}
