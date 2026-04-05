# AO Labs — Jekyll Personal Website

Personal website at `aolabs.dev` built on the Forty Jekyll theme with a Material
Ocean palette and a custom tile-first content UI.

## Project Layout

| Path                            | Purpose                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| `_posts/`                       | Blog posts and project write-ups in Markdown                       |
| `_pages/`                       | Static pages such as home, blog, projects, tags, search, and 404   |
| `_layouts/`                     | Page shells such as `post.html`, `home.html`, and archive views    |
| `_includes/`                    | Reusable partials for tags, search, navigation, tiles, and share   |
| `_sass/`                        | Sass source split into `base/`, `components/`, `layout/`, `libs/`  |
| `assets/css/main.scss`          | The single Sass entrypoint compiled by Jekyll                      |
| `assets/js/`                    | Frontend behavior for nav, search, share, Mermaid, and theme JS    |
| `assets/images/`                | Post and project imagery                                           |
| `scripts/`                      | Maintenance helpers such as tag generation and notebook conversion |
| `tags/`                         | Auto-generated tag pages; never edit manually                      |
| `scripts/update_tags.py`        | Regenerates `tags/*.md` from post front matter                     |
| `scripts/notebook_converter.py` | Converts staged notebooks into `_posts/` Markdown                  |

## Build

```bash
bundle exec jekyll serve  # local dev server
bundle exec jekyll build  # production build
```

## Working Conventions

- Permalinks: blog posts use `/blog/:title/`, project posts use
  `/projects/:title/`
- Post filenames: `YYYY-MM-DD-slug.md` in `_posts/`
- Tags: kebab-case only, and run `python scripts/update_tags.py` after tag edits
- Images: prefer `assets/images/<post-slug>/filename.ext` for post-specific
  assets
- Styles: keep edits in `_sass/**` and `assets/css/main.scss`; do not
  reintroduce deleted standalone stylesheets such as `extra.css`
- Navigation and search behavior are coordinated across
  `_includes/header.liquid`, `assets/js/main.js`, and the header/search Sass
  partials
- Mermaid rendering is handled by `_includes/mermaid.html`,
  `assets/js/mermaid-viewer.js`, and `_sass/components/_mermaid.scss`
- The repo does not currently track a `notebooks/` directory; create it first if
  you intend to use `scripts/notebook_converter.py`
- Commit messages should use Conventional Commits, preferably
  `type(scope): summary`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution philosophy.
