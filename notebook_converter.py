#!/usr/bin/env python

'''
notebook_converter.py

Copyright 2021 Abubakar Yagoub
Contact: blacksuan19.tk

This script converts a jupyter notebook to markdown
and moves converted markdown to posts and images to assets.
Requirements:
- jupyter (for converting notebooks to md)
- imagemagick (for adding background to images)
'''

import glob
import os
from datetime import datetime

post_dir = "_posts/"
notebooks_dir = "notebooks/"
assets_dir = "assets/images/"
current_date = datetime.today().strftime('%Y-%m-%d')
front_matter = """---
title: {}
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

# get all notebook files
filenames = glob.glob(notebooks_dir + '*.ipynb')


def check_exists(file):
    """check if a post from a notebook has already been created"""
    for post in glob.glob(post_dir + "/*.md"):
        if file in post:
            return True
    return False


for file in filenames:
    # get filename without directory prefix and extension
    name = file.split("/")[1].split(".")[0]
    new_name = f"{notebooks_dir}{current_date}-{name}.md"
    if check_exists(name):
        print(f"post for {name} has already been created.")
        continue

    os.system(f"jupyter nbconvert --to markdown {file}")

    # rename with date
    os.system(f"mv {notebooks_dir}{name}.md {new_name}")

    f = open(new_name, "r+")
    content = f.readlines()
    for i, line in enumerate(content):
        # fix assets path
        content[i] = content[i].replace("![png](", "![png](/assets/images/")
    f.seek(0)
    f.write(front_matter.format(name.replace("-", " ").title(), name))
    f.writelines(content)
    f.close()

    # add background color so text is visible in images
    os.system(
        f"mogrify -background white -flatten {notebooks_dir}{name}_files/*")

    # move file and assets
    os.system(f"mv {new_name} {post_dir}")
    os.system(f"rm -rf {assets_dir}{name}_files")
    os.system(f"mv {notebooks_dir}{name}_files {assets_dir}")
