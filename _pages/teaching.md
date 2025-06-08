---
layout: page
permalink: /teaching/
title: teaching
description: Materials for courses I've taught.
nav: false
nav_order: 6
---

<div class="projects">
  {% assign sorted_teaching = site.teaching | sort: "importance" %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_teaching %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
</div>
