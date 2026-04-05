---
description:
  "Audit, clean up, rename, or synchronize tags across all posts on this Jekyll
  site. Finds inconsistencies, duplicate variants, and missing tag pages, then
  runs scripts/update_tags.py."
agent: agent
---

Audit and update tags across all `_posts/*.md` files in this Jekyll site.

## Instructions

1. **Scan all posts** — read every file in `_posts/` and extract the `tags:`
   list from the front matter.

2. **Report findings**:
   - Full list of all unique tags and how many posts use each
   - Any tags that look like duplicates or variants (e.g. `ML` vs
     `machine-learning`, `NLP` vs `nlp`)
   - Any tags that are not kebab-case (e.g. `DataScience`, `Web Development`)
   - Posts missing a `tags:` field entirely

3. **Ask for confirmation** before making any changes.

4. **Apply fixes** — based on user input:
   - Rename tags to kebab-case consistently across all affected posts
   - Merge duplicate tag variants
   - Add missing tags to posts

5. **Regenerate tag pages** after all edits:

   ```bash
   python scripts/update_tags.py
   ```

6. **Confirm** the final tag count and list.

## Rules

- Tag names must be kebab-case, all lowercase: `data-science`, not `DataScience`
- The `tags/` directory is auto-generated — never edit it directly
- A tag page is only created if at least one post uses that tag
- Visual rendering is handled by `_includes/tag-cloud.liquid`,
  `_includes/post-tags.liquid`, `_includes/tag-chip.liquid`, and
  `_includes/tag-label.liquid`; keep content changes in post front matter and
  regenerate pages rather than editing tag pages directly
