---
layout: page
title: About
permalink: /about/
---

This is the base Jekyll theme. You can find out more info about customizing your Jekyll theme, as well as basic Jekyll usage documentation at [jekyllrb.com](https://jekyllrb.com/)

You can find the source code for Minima at GitHub:
[jekyll][jekyll-organization] /
[minima](https://github.com/jekyll/minima)

You can find the source code for Jekyll at GitHub:
[jekyll][jekyll-organization] /
[jekyll](https://github.com/jekyll/jekyll)


[jekyll-organization]: https://github.com/jekyll

<h1 class="col-header dark-orange">All posts</h1>
{% for post in site.posts %}
<div class="post-preview">
  <img class="post-preview__left" src="{{ post.image }}" alt="{{ page.image_alt }}">
  <div class="post-preview__right">
    <a class="preview-title" href="{{ post.url }}">{{ post.title }}</a>
    <span>{{ post.date | date: "%b %d, %Y" }}</span>
    <div class="tag-group">
      {% for tag in post.tags %}
        <div class="tag"><span class="tag-text">{{ tag }}</span></div>
      {% endfor %}
    </div>
  </div>
</div>
{% endfor %}