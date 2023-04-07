---
title: "Pytorch Object Detection"
layout: archive
permalink: /pytorchOD
author_profile: true
sidebar_main: true
sidebar:
    nav: "sidebar-category"
---


{% assign posts = site.categories.pytorchCV %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
