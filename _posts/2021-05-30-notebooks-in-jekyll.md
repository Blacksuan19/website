---
layout: post
title: Jupyter Notebooks in Jekyll
description: Converting Jupyter notebooks to jekyll posts
image: /assets/images/jupyter.jpeg
project: false
permalink: /blog/:title/
tags:
  - python
  - auto
  - data-science
---

jupyter notebooks are a very important instrument in a data scientists arsenal,
they allow quick prototyping and testing without having to set up a development
environment, with cloud services such as google colab and kaggle its much easier
to start experimenting with exploratory data analysis, machine learning and deep
learning, another advantage jupyter notebooks have is their form
interoperability, notebooks can be converted to HTML, pdf and my favorite
markdown.

most of my data science work these days is done with notebooks on kaggle or
Google colab, i have finished entire projects on just a notebook (don't judge
me), through time i have made a collection of those notebooks and now that i
have time to work on side projects why not add them to my personal website?

at first i searched for ways to embed a pdf in jekyll and found few ways but
none of which seemed like a good option because pdf's cannot be styled, next
option was just convert to html, that worked but again faced a similar issue,
jupyter exported html has its own css making it hard to override globally, this
website is built using Jekyll which uses markdown as the primary form of posts
(projects are just posts with project set to true) so i settled on markdown in
the end.

> why spend 1 hour converting few notebooks when i can spend few hours
> automating the whole process.

i decided to just script it, the initial steps are very clear

- create a folder to keep all notebook files (make sure to add it to
  .gitignore).
- go through all notebooks and convert them to markdown.
- move all markdown files to `_posts/`
- move all assets to site assets

```python
for file in filenames:
    # get filename without directory prefix and extension
    name = file.split("/")[1].split(".")[0]
    new_name = f"{notebooks_dir}{current_date}-{name}.md"
    # convert
    os.system(f"jupyter nbconvert --to markdown {file}")

    # rename with date
    os.system(f"mv {notebooks_dir}{name}.md {new_name}")

    # move file and assets
    os.system(f"mv {new_name} {post_dir}")
    os.system(f"rm -rf {assets_dir}{name}_files")
    os.system(f"mv {notebooks_dir}{name}_files {assets_dir}")
```

## Hiccups

- asset images path needs to be updated in each post
  - open the post
  - add front matter
  - update assets path
  - paste the rest of the post

```python
f = open(new_name, "r+")
content = f.readlines()
for i, line in enumerate(content):
    # fix assets path
    content[i] = content[i].replace("![png](", "![png](/assets/images/")
f.seek(0)
f.write(front_matter.format(name.replace("-", " ").title()))
f.writelines(content)
f.close()
```

- prevent regenerating posts for notebooks that have already been generated, the
  conversion might take time with large notebooks so we want to prevent doing it
  twice for any of the notebooks, we check the posts directory if there is any
  file with a name that contains the current notebook name.

```python
def check_exists(file):
    """check if a post from a notebook has already been created"""
    for post in glob.glob(post_dir + "/*.md"):
        if file in post:
            return True
    return False
```

- fix transparent image backgrounds, most graphs have a transparent background
  which is not ideal on the web specially on a site with a dark background,
  imagemagick can easily add a background to any image and remove its
  transparency

```python
# add background color so text is visible in images
os.system(f"mogrify -background white -flatten {notebooks_dir}{name}_files/*")
```

and with that everything is set, the script will do all the work for us, what is
left is just to push the changes (not lazy enough to automate that yet). example
[results]({% post_url 2021-05-24-sign-language-classification-with-pytorch-94 %}).

### Full script

```python
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
source:
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
    f.write(front_matter.format(name.replace("-", " ").title()))
    f.writelines(content)
    f.close()

    # add background color so text is visible in images
    os.system(
        f"mogrify -background white -flatten {notebooks_dir}{name}_files/*")

    # move file and assets
    os.system(f"mv {new_name} {post_dir}")
    os.system(f"rm -rf {assets_dir}{name}_files")
    os.system(f"mv {notebooks_dir}{name}_files {assets_dir}")
```
