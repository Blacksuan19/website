---
description:
  "Convert a Jupyter notebook into a Jekyll project post using
  scripts/notebook_converter.py. Stages the notebook, runs the converter, and
  updates the generated post metadata."
argument-hint: "Notebook name or path"
agent: agent
---

Convert a Jupyter notebook into a Jekyll project post for this website.

This mirrors what `scripts/notebook_converter.py` does automatically, but lets
you customise the output before writing the file.

## Instructions

1. Determine the notebook to convert — use the argument provided or ask for the
   `.ipynb` path.

2. Ensure the repository has a `notebooks/` staging directory because the
   converter script reads every `*.ipynb` from that location. If the directory
   is missing, create it first and place only the intended notebook there.

3. Check that `notebooks/<name>.ipynb` exists. If the user provided a notebook
   from another path, copy or move it into `notebooks/` before running the
   script.

4. Collect any missing details:
   - **Title** — defaults to the notebook stem with `-` replaced by spaces and
     title-cased
   - **Description** — one sentence summary (default: "Data Science Project")
   - **Kaggle source URL** — defaults to
     `https://www.kaggle.com/abubakaryagob/<name>`
   - **Extra tags** — in addition to the defaults: `data-science`,
     `machine-learning`, `project`

5. Run the converter script after confirming no unrelated notebooks are staged
   in `notebooks/` because the script converts every notebook in that folder:

   ```bash
   python scripts/notebook_converter.py
   ```

6. After the script runs, open the generated file in `_posts/` and apply any
   customisations to the front matter (title, description, source, tags).

7. Regenerate tag pages if new tags were added:

   ```bash
   python scripts/update_tags.py
   ```

8. Remind me to preview the converted post locally with
   `bundle exec jekyll serve`.

## Generated Front Matter Template

```yaml
---
title: Notebook Title
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/<name>
tags:
  - data-science
  - machine-learning
  - project
---
```

## Notes

- Image references in the notebook output use `/assets/images/<name>_files/`
  paths — the script handles this automatically
- The repository may not already contain a `notebooks/` directory; the workflow
  should create it when needed
- `scripts/notebook_converter.py` processes all staged notebooks in
  `notebooks/`, not just one target file
- If `imagemagick` is not installed, the background-flattening step (`mogrify`)
  will fail silently — install with `sudo apt install imagemagick`
- `jupyter` must be available in the current environment for `nbconvert` to work
