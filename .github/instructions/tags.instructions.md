---
description:
  "Use when adding, removing, or auditing tags on blog posts or project posts.
  Covers the full tags workflow: front matter format, running
  scripts/update_tags.py, and understanding auto-generated tag pages."
applyTo: "tags/**"
---

# Tags Workflow

## Rule: Never Edit `tags/` Manually

All files inside `tags/` are **auto-generated** by `scripts/update_tags.py`. Any
manual changes will be overwritten the next time the script runs.

## Adding Tags to a Post

In the post's YAML front matter, tags are a list under the `tags` key:

```yaml
tags:
  - python
  - data-science
  - machine-learning
```

### Tag Naming Rules

- Always **kebab-case**: `data-science`, not `DataScience` or `data science`
- All lowercase (the tag cloud capitalizes them at display time)
- Be consistent with existing tags — reuse the same tag string rather than
  creating variants

## Regenerating Tag Pages

After adding, removing, or renaming any tag in any post, run:

```bash
python scripts/update_tags.py
```

This will:

1. Scan all `_posts/*.md` front matter for `tags:` lists
2. Delete all existing `tags/*.md` files
3. Recreate one `tags/<tag>.md` per unique tag

Each generated file looks like:

```yaml
---
layout: tag-page
title: "Tag: data-science"
tag: data-science
robots: noindex
---
```

## Tag Display

- `_pages/tags.html` — the `/tags/` archive page
- `_includes/tag-cloud.liquid` — renders the full tag archive using reusable
  chips
- `_includes/post-tags.liquid` — renders post tag stacks with overflow handling
- `_includes/tag-chip.liquid` — shared tag badge component
- `_includes/tag-label.liquid` — display formatter for common acronyms and
  labels

These files handle display automatically — no manual tag page edits are needed
when adding tags.

## Checking Existing Tags

To see all current tags:

```bash
ls tags/
```

Or look at the tag cloud at `/tags/` in the dev server.

## Bulk Tag Audit

To list all tags across all posts (before running the script):

```bash
grep -h "^  - " _posts/*.md | sort -u
```
