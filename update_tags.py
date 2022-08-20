#!/usr/bin/env python

"""
tag_generator.py

Copyright 2017 Long Qian
Contact: lqian8@jhu.edu

This script creates tags for your Jekyll blog hosted by Github page.
No plugins required.
"""

from pathlib import Path

post_dir = Path("_posts/")
tag_dir = Path("tags/")

filenames = list(post_dir.glob("*.md"))

total_tags = []
for filename in filenames:
    f = open(filename, "r", encoding="utf8")
    crawl = False
    for line in f:
        # check if we are in the front matter
        if line.strip() == "---":
            if not crawl:
                crawl = True
            else:
                crawl = False
                break
        if crawl:
            current_tags = line.strip().split()
            if current_tags[0] == "-":
                total_tags.extend(current_tags[1:])

    f.close()
total_tags = set(total_tags)

old_tags = list(tag_dir.glob("*.md"))
list(map(lambda tag: tag.unlink(missing_ok=True), old_tags))

tag_dir.mkdir(exist_ok=True)

for tag in total_tags:
    tag_filename = tag_dir / Path(f"{tag}.md")
    write_str = (
        '---\nlayout: tag-page\ntitle: "Tag: '
        + tag
        + '"\ntag: '
        + tag
        + "\nrobots: noindex\n---\n"
    )
    tag_filename.write_text(write_str)
print("Tags generated, count", len(total_tags))
