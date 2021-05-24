# convert a jupyter notebook to markdown and move it and its files to the appropriate place
import glob
import os
from datetime import datetime

post_dir = "_posts/"
notebooks_dir = "notebooks/"
assets_dir = "assets/images/"
current_date = datetime.today().strftime('%Y-%m-%d')

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

    front_matter = f"""---
title: {name.replace("-", " ").title()}
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

    # add front matter
    f = open(new_name, "r+")
    content = f.readlines()
    # fix assets path
    for i, line in enumerate(content):
        content[i] = content[i].replace(r"# *", "")
        content[i] = content[i].replace("![png](", "![png](/assets/images/")
    f.seek(0)
    f.write(front_matter)
    f.writelines(content)
    f.close()

    # move file and assets
    os.system(f"mv {new_name} {post_dir}")
    os.system(f"rm -rf {assets_dir}{name}_files")
    os.system(f"mv {notebooks_dir}{name}_files {assets_dir}")
