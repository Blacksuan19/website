#!/usr/bin/env python

"""
notebook_converter.py

Copyright 2021 Abubakar Yagoub
Contact: aolabs.dev

This script converts a jupyter notebook to markdown
and moves converted markdown to posts and images to assets.
Requirements:
- jupyter (for converting notebooks to md)
- imagemagick (for adding background to images)
"""

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
post_dir = REPO_ROOT / "_posts"
notebooks_dir = REPO_ROOT / "notebooks"
assets_dir = REPO_ROOT / "assets/images"
current_date = datetime.today().strftime("%Y-%m-%d")
front_matter = """---
title: {}
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/{}
tags:
  - data-science
  - machine-learning
  - project
---\n\n"""

filenames = list(notebooks_dir.glob("*.ipynb"))


def check_exists(file_name: str) -> bool:
    """Check if a post from a notebook has already been created."""
    for post in map(lambda file_path: file_path.name, post_dir.glob("*.md")):
        if file_name in post:
            return True
    return False


for file_path in filenames:
    name = file_path.stem
    new_name = notebooks_dir / f"{current_date}-{name}.md"
    if check_exists(name):
        print(f"post for {name} has already been created.")
        continue

    os.system(f'jupyter nbconvert --to markdown "{file_path}"')
    os.system(f'mv "{notebooks_dir / (name + ".md")}" "{new_name}"')

    with open(new_name, "r+", encoding="utf8") as file_handle:
        content = file_handle.readlines()
        for index, line in enumerate(content):
            content[index] = line.replace("![png](", "![png](/assets/images/")
        file_handle.seek(0)
        file_handle.write(front_matter.format(name.replace("-", " ").title(), name))
        file_handle.writelines(content)

    os.system(
        f'mogrify -background white -flatten "{notebooks_dir / (name + "_files")}"/*'
    )
    os.system(f'mv "{new_name}" "{post_dir}"')
    os.system(f'rm -rf "{assets_dir / (name + "_files")}"')
    os.system(f'mv "{notebooks_dir / (name + "_files")}" "{assets_dir}"')
