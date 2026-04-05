---
description:
  "Use when creating blog posts, project write-ups, or any new post. Covers
  front matter fields, file naming, permalink patterns, and the tags workflow
  for Jekyll _posts/ on this site."
---

# Post Creation Guidelines

## File Naming

All posts live in `_posts/` and must follow: `YYYY-MM-DD-kebab-case-slug.md`

Example using the current date: `2026-04-04-my-new-post.md`

## Front Matter

### Blog Post (`project: false`)

```yaml
---
title: Human Readable Title
layout: post
description: One-sentence summary shown in tiles and SEO
image: /assets/images/post-slug/filename.jpg
project: false
permalink: "/blog/:title/"
tags:
  - tag-one
  - tag-two
---
```

### Project Post (`project: true`)

```yaml
---
title: Project Name
layout: post
description: What the project does
image: /assets/images/project-slug/filename.jpg
project: true
permalink: "/projects/:title/"
source: https://github.com/blacksuan19/repo # optional — shows a code button
tags:
  - project
  - python
  - data-science
---
```

## Field Reference

| Field         | Required | Notes                                                                     |
| ------------- | -------- | ------------------------------------------------------------------------- |
| `title`       | Yes      | Display name used in nav and tiles                                        |
| `layout`      | Yes      | Always `post`                                                             |
| `description` | Yes      | Shown in tiles, search results, and meta tags                             |
| `image`       | Yes      | Featured image; store in `assets/images/<post-slug>/` when practical      |
| `project`     | Yes      | `true` → projects feed; `false` → blog feed                               |
| `permalink`   | Yes      | `/blog/:title/` or `/projects/:title/`                                    |
| `tags`        | Yes      | YAML list, kebab-case, run `scripts/update_tags.py` after adding new ones |
| `source`      | No       | GitHub/Kaggle URL — renders a code icon on the post header                |

## Tags

- Always kebab-case: `data-science`, `web-development`, `machine-learning`
- After editing tags in any post, regenerate tag pages:
  ```bash
  python scripts/update_tags.py
  ```
- The `tags/` directory is entirely auto-generated — **never edit it manually**

## Images

- Store under `assets/images/<post-slug>/` for post-specific assets
- Reference in front matter as `/assets/images/<post-slug>/filename.ext`
- For data science posts without a custom image, the default `ds.jpg` can be
  reused

## After Creating a Post

1. Run `python scripts/update_tags.py` if new tags were added
2. Verify locally with `bundle exec jekyll serve`
3. Check the tile appears on the correct feed and the tag chips render correctly
