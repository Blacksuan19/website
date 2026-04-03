#!/usr/bin/env python

"""
tag_generator.py

Copyright 2017 Long Qian
Contact: lqian8@jhu.edu

This script creates tags for your Jekyll blog hosted by Github page.
No plugins required.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
post_dir = REPO_ROOT / "_posts"
tag_dir = REPO_ROOT / "tags"

filenames = list(post_dir.glob("*.md"))

total_tags = []
for filename in filenames:
    with open(filename, "r", encoding="utf8") as file_handle:
        crawl = False
        for line in file_handle:
            if line.strip() == "---":
                if not crawl:
                    crawl = True
                else:
                    crawl = False
                    break
            if crawl:
                current_tags = line.strip().split()
                if current_tags and current_tags[0] == "-":
                    total_tags.extend(current_tags[1:])

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
    tag_filename.write_text(write_str, encoding="utf8")

print("Tags generated, count", len(total_tags))
