#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Palette:
    name: str
    accent_a: str
    accent_b: str
    glow: str


@dataclass(frozen=True)
class CoverMeta:
    slug: str
    title: str
    description: str
    tags: list[str]
    image: str


PALETTES = [
    Palette("ocean", "#6fc3df", "#8ea9e8", "#6fc3df"),
    Palette("mint", "#87c5a4", "#6fc3df", "#87c5a4"),
    Palette("violet", "#8d82c4", "#8ea9e8", "#8d82c4"),
    Palette("coral", "#ec8d81", "#e7b788", "#ec8d81"),
    Palette("cyan", "#9bf1ff", "#6fc3df", "#9bf1ff"),
    Palette("sunset", "#e7b788", "#ec8d81", "#e7b788"),
    Palette("twilight", "#8d82c4", "#ec8d81", "#8d82c4"),
]

PALETTES_BY_NAME = {palette.name: palette for palette in PALETTES}

WIDE_CHARS = set("WMQGOD@#%&")
NARROW_CHARS = set("ilI.,:;'!|[]() ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SVG covers for project posts by reading post frontmatter and "
            "wrapping title text to fit the existing Forty layout."
        )
    )
    parser.add_argument("posts", nargs="*", help="Path(s) to post markdown files")
    parser.add_argument(
        "--palette",
        choices=sorted(PALETTES_BY_NAME),
        help="Use a specific palette group instead of random selection",
    )
    parser.add_argument(
        "--accent-a",
        help="Override the primary accent color with a hex value",
    )
    parser.add_argument(
        "--accent-b",
        help="Override the secondary accent color with a hex value",
    )
    parser.add_argument(
        "--glow",
        help="Override the glow color with a hex value",
    )
    parser.add_argument(
        "--output",
        help="Write a single output file to this path. Only valid when generating one post.",
    )
    parser.add_argument(
        "--update-post",
        action="store_true",
        help="Update the post frontmatter image field to the generated cover path",
    )
    parser.add_argument(
        "--list-palettes",
        action="store_true",
        help="Print available palette names and exit",
    )
    parser.add_argument(
        "--snapshot-label",
        default="project snapshot",
        help="Footer label shown in the snapshot card",
    )
    parser.add_argument(
        "--card-title",
        default="Snapshot",
        help="Top title shown inside the snapshot card",
    )
    parser.add_argument(
        "--hero-title",
        default=None,
        help="Primary title shown on the left side of the cover",
    )
    parser.add_argument(
        "--hero-subtitle",
        default=None,
        help="Secondary subtitle shown below the primary title",
    )
    parser.add_argument(
        "--chip",
        action="append",
        default=None,
        help="Explicit chip label to render. Repeat up to three times.",
    )
    return parser.parse_args()


def estimate_text_width(text: str, font_size: int) -> float:
    units = 0.0
    for char in text:
        if char in WIDE_CHARS:
            units += 0.92
        elif char in NARROW_CHARS:
            units += 0.34
        elif char.isupper():
            units += 0.72
        else:
            units += 0.58
    return units * font_size


def wrap_text(text: str, font_size: int, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if estimate_text_width(candidate, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_text_block(
    text: str,
    initial_font_size: int,
    min_font_size: int,
    max_width: int,
    max_lines: int,
) -> tuple[list[str], int]:
    if not text:
        return [], initial_font_size

    for font_size in range(initial_font_size, min_font_size - 1, -2):
        lines = wrap_text(text, font_size, max_width)
        if len(lines) <= max_lines:
            return lines, font_size

    lines = wrap_text(text, min_font_size, max_width)
    if len(lines) > max_lines:
        clipped = lines[:max_lines]
        last_line = clipped[-1]
        while (
            last_line
            and estimate_text_width(f"{last_line}...", min_font_size) > max_width
        ):
            last_line = (
                last_line.rsplit(" ", 1)[0] if " " in last_line else last_line[:-1]
            )
        clipped[-1] = f"{last_line}..." if last_line else "..."
        return clipped, min_font_size
    return lines, min_font_size


def fit_single_line(
    text: str, initial_font_size: int, min_font_size: int, max_width: int
) -> int:
    for font_size in range(initial_font_size, min_font_size - 1, -1):
        if estimate_text_width(text, font_size) <= max_width:
            return font_size
    return min_font_size


def pick_chips(tags: list[str], chip_override: list[str] | None = None) -> list[str]:
    if chip_override:
        chips = chip_override[:3]
        while len(chips) < 3:
            chips.append("project")
        return chips

    meaningful = [tag.replace("-", " ") for tag in tags if tag != "project"]
    chips = meaningful[:3]
    fallbacks = ["analysis", "engineering", "project"]
    while len(chips) < 3:
        chips.append(fallbacks[len(chips)])
    return chips


def parse_frontmatter(post_path: Path) -> CoverMeta:
    text = post_path.read_text(encoding="utf8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit(f"No frontmatter found in {post_path}")

    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", post_path.stem)
    frontmatter = match.group(1).splitlines()

    data: dict[str, str | list[str]] = {}
    index = 0
    while index < len(frontmatter):
        line = frontmatter[index]
        key_match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not key_match:
            index += 1
            continue

        key = key_match.group(1)
        raw_value = key_match.group(2).strip()

        if key == "tags":
            tags: list[str] = []
            index += 1
            while index < len(frontmatter):
                tag_match = re.match(r"^\s*-\s*(.+?)\s*$", frontmatter[index])
                if not tag_match:
                    break
                tags.append(tag_match.group(1).strip().strip('"').strip("'"))
                index += 1
            data[key] = tags
            continue

        if raw_value:
            data[key] = raw_value.strip('"').strip("'")
            index += 1
            continue

        collected: list[str] = []
        index += 1
        while index < len(frontmatter):
            child = frontmatter[index]
            if child.startswith("  ") and not re.match(r"^\s*-\s+", child):
                collected.append(child.strip().strip('"').strip("'"))
                index += 1
                continue
            break
        data[key] = " ".join(part for part in collected if part)

    return CoverMeta(
        slug=slug,
        title=str(data.get("title", slug.replace("-", " ").title())),
        description=str(data.get("description", "")),
        tags=list(data.get("tags", [])),
        image=str(data.get("image", "")),
    )


def resolve_post_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Post not found: {path}")
    return path


def resolve_output_path(meta: CoverMeta, output_override: str | None) -> Path:
    if output_override:
        output = Path(output_override)
        if not output.is_absolute():
            output = REPO_ROOT / output
        return output.resolve()

    if meta.image.startswith("/assets/images/"):
        image_rel = meta.image.removeprefix("/assets/images/")
        image_dir = Path(image_rel).parent
        if str(image_dir) != ".":
            return (REPO_ROOT / "assets" / "images" / image_dir / "cover.svg").resolve()

    return (REPO_ROOT / "assets" / "images" / meta.slug / "cover.svg").resolve()


def web_path_for(output_path: Path) -> str:
    return "/" + output_path.relative_to(REPO_ROOT).as_posix()


def update_post_image(post_path: Path, new_image: str) -> None:
    lines = post_path.read_text(encoding="utf8").splitlines()

    start = None
    end = None
    for index, line in enumerate(lines):
        if line.strip() == "---":
            if start is None:
                start = index
            else:
                end = index
                break

    if start is None or end is None:
        raise SystemExit(f"No frontmatter found in {post_path}")

    for index in range(start + 1, end):
        if lines[index].startswith("image:"):
            lines[index] = f"image: {new_image}"
            break
    else:
        lines.insert(end, f"image: {new_image}")

    post_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def tspan_lines(lines: Iterable[str], x: int, line_height: int) -> str:
    escaped = [html.escape(line) for line in lines if line]
    if not escaped:
        return ""
    chunks = [f'<tspan x="{x}" dy="0">{escaped[0]}</tspan>']
    for line in escaped[1:]:
        chunks.append(f'<tspan x="{x}" dy="{line_height}">{line}</tspan>')
    return "\n    ".join(chunks)


def chip_group(chips: list[str], y: int) -> str:
    parts: list[str] = []
    x = 120
    font_size = 30
    pill_height = 48
    for chip in chips:
        label = html.escape(chip)
        width = int(estimate_text_width(chip, font_size) + 52)
        center_x = width // 2
        parts.append(
            f"""<g transform="translate({x} {y})">
    <rect width="{width}" height="{pill_height}" rx="24" fill="#ffffff" fill-opacity="0.06" stroke="#d6e2ff" stroke-opacity="0.18"/>
    <text x="{center_x}" y="28" text-anchor="middle" dominant-baseline="middle" fill="#ffffff" fill-opacity="0.88" font-family="Arial, Helvetica, sans-serif" font-size="{font_size}">{label}</text>
  </g>"""
        )
        x += width + 22
    return "\n  ".join(parts)


def generate_svg(
    meta: CoverMeta,
    palette: Palette,
    snapshot_label: str,
    card_title: str,
    hero_title: str | None,
    hero_subtitle: str | None,
    chip_override: list[str] | None,
) -> str:
    primary_title = (hero_title or meta.title).strip()
    secondary_title = (hero_subtitle or "").strip()

    title_font_size = fit_single_line(primary_title, 98, 76, 620)
    title_lines = [primary_title]
    title_line_height = int(title_font_size * 0.92)
    title_top = 305
    chips_top = 560

    title_block = tspan_lines(title_lines, 120, title_line_height)
    chips_block = chip_group(pick_chips(meta.tags, chip_override), chips_top)
    subtitle_block = ""
    if secondary_title:
        subtitle_lines, subtitle_font_size = fit_text_block(
            secondary_title,
            initial_font_size=38,
            min_font_size=30,
            max_width=820,
            max_lines=3,
        )
        subtitle_line_height = int(subtitle_font_size * 1.36)
        subtitle_block = f"""  <text x="120" y="390" fill="#d6e2ff" font-family="Arial, Helvetica, sans-serif" font-size="{subtitle_font_size}" font-weight="600">
    {tspan_lines(subtitle_lines, 120, subtitle_line_height)}
  </text>
"""
    card_snapshot_label = html.escape(snapshot_label)
    card_title_text = html.escape(card_title)
    card_title_font_size = fit_single_line(card_title, 26, 20, 240)
    footer_width = min(max(int(estimate_text_width(snapshot_label, 18) + 56), 220), 272)
    footer_x = 34 + ((292 - footer_width) // 2)
    footer_text_x = footer_x + (footer_width // 2)
    footer_font_size = fit_single_line(snapshot_label, 20, 17, footer_width - 24)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" role="img">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f111a"/>
      <stop offset="100%" stop-color="#181b26"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{palette.accent_a}"/>
      <stop offset="100%" stop-color="{palette.accent_b}"/>
    </linearGradient>
  </defs>
  <rect width="1600" height="900" fill="url(#bg)"/>
    <circle cx="1370" cy="160" r="200" fill="{palette.glow}" fill-opacity="0.09"/>
    <circle cx="1220" cy="780" r="170" fill="#ec8d81" fill-opacity="0.08"/>
    <circle cx="260" cy="760" r="150" fill="#8d82c4" fill-opacity="0.10"/>
  <rect x="120" y="130" width="250" height="44" rx="22" fill="#9bf1ff" fill-opacity="0.14" stroke="#9bf1ff" stroke-opacity="0.5"/>
  <text x="145" y="159" fill="#9bf1ff" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700">AO Labs Project</text>
  <text x="120" y="{title_top}" fill="#ffffff" font-family="Arial, Helvetica, sans-serif" font-size="{title_font_size}" font-weight="800">
    {title_block}
  </text>
{subtitle_block}  
  {chips_block}
    <g transform="translate(1140 245)">
        <rect width="360" height="410" rx="30" fill="#ffffff" fill-opacity="0.05" stroke="#9bf1ff" stroke-opacity="0.4"/>
        <rect x="34" y="34" width="292" height="52" rx="14" fill="#ffffff" fill-opacity="0.08"/>
        <text x="180" y="68" text-anchor="middle" fill="{palette.accent_a}" font-family="Arial, Helvetica, sans-serif" font-size="{card_title_font_size}" font-weight="700">{card_title_text}</text>
        <rect x="34" y="116" width="184" height="18" rx="9" fill="#ffffff" fill-opacity="0.86"/>
        <rect x="34" y="152" width="132" height="18" rx="9" fill="#ffffff" fill-opacity="0.58"/>
        <rect x="34" y="188" width="210" height="18" rx="9" fill="#ffffff" fill-opacity="0.86"/>
        <rect x="34" y="224" width="156" height="18" rx="9" fill="#ffffff" fill-opacity="0.58"/>
        <path d="M34 292 C94 252, 154 252, 214 292 S334 332, 326 286" fill="none" stroke="url(#accent)" stroke-width="10" stroke-linecap="round"/>
                <rect x="{footer_x}" y="348" width="{footer_width}" height="40" rx="20" fill="{palette.accent_b}" fill-opacity="0.22" stroke="{palette.accent_b}" stroke-opacity="0.45"/>
                <text x="{footer_text_x}" y="374" text-anchor="middle" fill="#e5faff" font-family="Arial, Helvetica, sans-serif" font-size="{footer_font_size}" font-weight="700">{card_snapshot_label}</text>
  </g>
</svg>
"""


def generate_cover(
    post_path: Path,
    palette: Palette,
    output_override: str | None,
    update_post: bool,
    snapshot_label: str,
    card_title: str,
    hero_title: str | None,
    hero_subtitle: str | None,
    chip_override: list[str] | None,
) -> None:
    meta = parse_frontmatter(post_path)
    output_path = resolve_output_path(meta, output_override)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_svg(
            meta,
            palette,
            snapshot_label,
            card_title,
            hero_title,
            hero_subtitle,
            chip_override,
        ),
        encoding="utf8",
    )

    web_path = web_path_for(output_path)
    if update_post:
        update_post_image(post_path, web_path)

    print(f"Generated {web_path} using palette {palette.name} for {meta.title}")


def main() -> None:
    args = parse_args()

    if args.list_palettes:
        for palette in PALETTES:
            print(palette.name)
        return

    if not args.posts:
        raise SystemExit("At least one post path is required")

    if args.output and len(args.posts) != 1:
        raise SystemExit("--output can only be used with a single post")

    custom_palette = None
    if args.accent_a or args.accent_b or args.glow:
        if not (args.accent_a and args.accent_b):
            raise SystemExit("--accent-a and --accent-b must be provided together")
        custom_palette = Palette(
            "custom",
            args.accent_a,
            args.accent_b,
            args.glow or args.accent_a,
        )

    for raw_post in args.posts:
        post_path = resolve_post_path(raw_post)
        palette = custom_palette or (
            PALETTES_BY_NAME[args.palette] if args.palette else random.choice(PALETTES)
        )
        generate_cover(
            post_path,
            palette,
            args.output,
            args.update_post,
            args.snapshot_label,
            args.card_title,
            args.hero_title,
            args.hero_subtitle,
            args.chip,
        )


if __name__ == "__main__":
    main()
