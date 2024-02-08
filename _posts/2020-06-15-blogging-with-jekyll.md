---
category: tools
---

This blog is written using [Jekyll](https://jekyllrb.com/) and 
[Github Pages](https://pages.github.com/). Jekyll is a static website generator, which converts
directories of markdown files into pretty, hyperlinked blogs. Github interprets any repository with the
name `[github-username].github.io` as a Jekyll blog. This means the process of making a blog ridiculously easy:

1. Install `jekyll`
2. Run `jekyll new [blog_name]`
3. Drop some markdown files into the `_posts` directory. They should have a filename following
`[year-month-day-title].md` (like `2018-08-29-bananas.md`), and should start with this YAML snippet:
```yaml
---
layout: post
---
```
4. Run `jekyll serve --watch` to look at your post locally on port 4000.
5. Push to `[github-username].github.io` for the world to see.

That's all there is to it!

# Themes

You might notice that the website Jekyll generates contains a number of things you never specified in
your markdown files. There's a front page that lists all your blog posts, a header on all pages
that shows the name of your blog, a footer with a convenient RSS subscription button, and some pretty CSS. Where did
these come from?

These extras are filled in from your 'theme'. You can set which theme you're using in the `config.yml` file
Jekyll has helpfully placed in your blog's directory. The default theme is called 'minima'.

You can see the extra files your theme is including for you
by running `bundle info --path [theme_name]`. For example:

```console
> tree $(bundle info --path minima)
├── assets
│   ├── main.scss
│   └── minima-social-icons.svg
├── _includes
│   ├── disqus_comments.html
│   ├── footer.html
│   ├── google-analytics.html
│   ├── header.html
│   ├── head.html
│   ├── icon-github.html
│   ├── icon-github.svg
│   ├── icon-twitter.html
│   ├── icon-twitter.svg
│   └── social.html
├── _layouts
│   ├── default.html
│   ├── home.html
│   ├── page.html
│   └── post.html
├── LICENSE.txt
├── README.md
└── _sass
    ├── minima
    │   ├── _base.scss
    │   ├── _layout.scss
    │   └── _syntax-highlighting.scss
└── minima.scss
```

## Templates

Take a look at one of the files in your theme directory.

```console
{% raw %}
> cat _layouts/page.html
---
layout: default
---
<article class="post">

  <header class="post-header">
    <h1 class="post-title">{{ page.title | escape }}</h1>
  </header>

  <div class="post-content">
    {{ content }}
  </div>

</article>
{% endraw %}
```

That's not normal html! What's going on?

The files in your theme's directory use a template language called Liquid.
Interpolation happens between pairs of curly brackets. When a file starts with a YAML block
containing a `layout` keyword, its body gets filled in as the `content` variable for associated template in the `_layouts` directory. 
Blog posts written with the YAML snippet above would use `_layouts/post.html`. But you don't need to stick to templates defined by your theme.
Make a `_layouts` sub-directory of your own! Any files you put here will also be valid layouts for your blog posts. 


## MathJax

You can take advantage of these custom layouts to add [MathJax](https://www.mathjax.org/) support to your blog. For example, I have a `_layouts/post.html`
file with the following content:

```html
{% raw %}
---
layout: default
---
<article class="post">
  <header class="post-header">
    <h1 class="post-title p-name" itemprop="name headline">{{ page.title | escape }}</h1>
    <p class="post-meta">
      <time class="dt-published" datetime="{{ page.date | date_to_xmlschema }}" itemprop="datePublished">
        {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
        {{ page.date | date: date_format }}
      </time>
    </p>
  </header>

  <div class="entry">
    {{ content }}
  </div>
  {% include mathjax.html %}
</article>
{% endraw %}
```

This is exactly the same as the default `_layouts/post.html` template, but I've added an instruction
to include mathjax as well. This tells every blog post that uses the `post` layout
to load the `mathjax.html` resource necessary for rendering Latex. 

Where do these included files come from? Well, we have to write those too. They live in the `_includes`
directory. Here's my `_includes/mathjax.html` file:

```html
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    }
  });
</script>
<script
  type="text/javascript"
  charset="utf-8"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
>
</script>
```

## Editors

While editing Markdown from a plain text editor isn't awful -- that was kind of the point of the language, after all -- 
other tools can make the experience a lot better. 
For most posts (indeed, for pretty much everything I write nowadays), I use
[Typora](https://typora.io/). It updates your rendered document as you type, which is especially convenient
for Latex expressions. For blog posts with code, I tend to use Jupyter notebooks.
