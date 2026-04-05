---
description:
  "Create a new project write-up post on aolabs.dev. Generates a Markdown file
  in _posts/ with project front matter (project: true), source link, and the
  projects permalink structure."
argument-hint: "Project name or description"
agent: agent
---

Create a new Jekyll project post for this website following the
[post creation guidelines](.github/instructions/post-creation.instructions.md).

## Instructions

1. Ask me (or use the argument provided) for:
   - **Title** — project name, e.g. "Credit Card Fraud Detection with PyTorch"
   - **Description** — one sentence describing what the project does
   - **Source URL** — GitHub or Kaggle link (optional; omit the field if not
     provided)
   - **Tags** — comma-separated; I'll convert to kebab-case (`data-science`,
     `python`, `machine-learning`, `project`, …)
   - **Image filename** — file in `assets/images/<project-slug>/` (default:
     `cover.jpg` or `cover.png` inside that folder)
   - **Content outline** — sections, results, methodology, or a full draft

2. Generate the post file:
   - Filename: `YYYY-MM-DD-slug.md` in `_posts/` using today's actual date
   - Use `layout: post`, `project: true`, `permalink: "/projects/:title/"`
   - Include `source:` field only if a URL was provided
   - Always include `project` in the tags list
   - Ensure `description` and `image` are present because project tiles depend
     on them

3. Create the file in `_posts/`

4. If any new tags were added, remind me to run:
   ```bash
   python scripts/update_tags.py
   ```
5. Remind me to preview the project locally with `bundle exec jekyll serve`.

## Example Output

```yaml
---
title: Credit Card Fraud Detection with PyTorch
layout: post
description:
  Building a binary classifier with PyTorch to detect fraudulent transactions.
image: /assets/images/credit-card-fraud-detection-with-pytorch/cover.jpg
project: true
permalink: "/projects/:title/"
source: https://www.kaggle.com/abubakaryagob/credit-card-fraud
tags:
  - data-science
  - machine-learning
  - python
  - project
---
Project content goes here...
```
