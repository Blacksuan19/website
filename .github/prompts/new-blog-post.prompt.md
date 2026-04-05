---
description:
  "Create a new blog post on aolabs.dev. Generates a properly structured
  Markdown file in _posts/ with correct front matter, kebab-case filename, and
  instructions to run scripts/update_tags.py."
argument-hint: "Topic or title of the new blog post"
agent: agent
---

Create a new Jekyll blog post for this website following the
[post creation guidelines](.github/instructions/post-creation.instructions.md).

## Instructions

1. Ask me (or use the argument provided) for:
   - **Title** — human-readable, e.g. "Setting Up a Dev Container in VS Code"
   - **Description** — one sentence shown in tiles and SEO
   - **Tags** — comma-separated list; I'll convert to kebab-case YAML list
   - **Image filename** — file in `assets/images/<post-slug>/` (or use a
     placeholder)
   - **Content summary** — brief outline or full draft

2. Generate the post file:
   - Filename: `YYYY-MM-DD-slug.md` in `_posts/` using today's actual date
   - Use `layout: post`, `project: false`, `permalink: "/blog/:title/"`
   - Tags in kebab-case YAML list format
   - Ensure `description` and `image` are present because archive tiles and
     search results depend on them

3. Create the file in `_posts/`

4. If any new tags were added that don't already exist in `tags/`, remind me to
   run:
   ```bash
   python scripts/update_tags.py
   ```
5. Remind me to preview the post locally with `bundle exec jekyll serve`.

## Example Output

```yaml
---
title: My New Post
layout: post
description: A short description of what this post covers.
image: /assets/images/my-new-post/post-image.jpg
project: false
permalink: "/blog/:title/"
tags:
  - linux
  - cli
  - setup
---
Post content goes here...
```
