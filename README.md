# AO Labs Personal Site

This repository contains my personal website built with Jekyll and based on a
heavily customized version of the Forty theme by
[HTML5 UP](https://html5up.net/). It has evolved well beyond the original
template into the version of the site I use for blog posts, project write-ups,
notes, and experiments.

<p align="center">
  <img src="/assets/images/forty.jpg" alt="Forty Theme">
</p>

## Site Highlights

- Material Ocean-inspired visual design
- separate blog and project archives
- tile-based landing, archive, and tag pages
- site-wide search powered by [lunr](https://lunrjs.com)
- compile-time syntax highlighting via [Shiki](https://shiki.style)
- tag archive pages generated from post front matter
- Mermaid diagram rendering support
- embeds for YouTube and Asciinema
- read-time metadata and post navigation
- RSS feed, sitemap, and SEO metadata
- contact form and social profile links

## Using Included Templates

### YouTube

Used to link to a YouTube video

```liquid
{% include youtube.liquid id="video_id" -%}
```

### Asciinema

Used to link to an asciinema video

```liquid
{% include asciinema.liquid id="video_id" -%}
```

## Project Structure

```bash
.
├── assets
│   ├── css
│   ├── fonts
│   ├── images
│   └── js
├── _includes
├── _layouts
├── _pages
├── _posts
├── _sass
│   ├── base
│   ├── components
│   ├── layout
│   └── libs
├── scripts
└── tags
```

## Local Development

- run `bundle install`
- run `npm install`
- run `bundle exec jekyll serve`
- open `http://localhost:4000`

Useful commands:

```bash
npm install
bundle exec jekyll serve
bundle exec jekyll build
python scripts/update_tags.py
python scripts/notebook_converter.py
```

## Content Notes

- posts live in `_posts/` and use `YYYY-MM-DD-slug.md`
- blog posts use `/blog/:title/`
- project posts use `/projects/:title/`
- tags should stay kebab-case
- `tags/` is generated content and should not be edited manually

## Credits

- [andrewbanchich](https://github.com/andrewbanchich/forty-jekyll-theme) -
  Ported original template to Jekyll.
- [Jekyll without plugins](https://jekyllcodex.org/without-plugins/) - Feed,
  Site map, SEO and other things.

Original README from HTML5 UP:

```md
Forty by HTML5 UP html5up.net | @ajlkn Free for personal and commercial use
under the CCA 3.0 license (html5up.net/license)

This is Forty, my latest and greatest addition to HTML5 UP and, per its
incredibly creative name, my 40th (woohoo)! It's built around a grid of "image
tiles" that are set up to smoothly transition to secondary landing pages (for
which a separate page template is provided), and includes a number of neat
effects (check out the menu!), extra features, and all the usual stuff you'd
expect. Hope you dig it!

Demo images\* courtesy of Unsplash, a radtastic collection of CC0 (public
domain) images you can use for pretty much whatever.

(\* = not included)

AJ aj@lkn.io | @ajlkn

Credits:

Demo Images: Unsplash (unsplash.com)

Icons: Font Awesome (fontawesome.github.com/Font-Awesome)

Other: jQuery (jquery.com) html5shiv.js (@afarkas @jdalton @jon_neal @rem)
background-size polyfill (github.com/louisremi) Misc. Sass functions
(@HugoGiraudel) Respond.js (j.mp/respondjs) Skel (skel.io)
```

Repository [Jekyll logo](https://github.com/jekyll/brand) icon licensed under a
[Creative Commons Attribution 4.0 International License](http://choosealicense.com/licenses/cc-by-4.0/).
