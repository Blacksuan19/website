#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "assets" / "images"
OPTIMIZED_DIR = IMAGES_DIR / "optimized"
MANIFEST_PATH = ROOT_DIR / "_data" / "optimized_images.json"
DEFAULT_WIDTHS = (320, 480, 640, 768, 960, 1280, 1600, 1920)
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_DIRS = {"optimized"}
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 4))


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed")

    return result.stdout.strip()


def identify_image(path: Path) -> tuple[int, int]:
    output = run_command(
        ["magick", "identify", "-ping", "-format", "%w %h", f"{path}[0]"]
    )
    width_text, height_text = output.split()
    return int(width_text), int(height_text)


def source_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def eligible_widths(original_width: int) -> list[int]:
    widths = [width for width in DEFAULT_WIDTHS if width < original_width]

    if not widths or widths[-1] != original_width:
        widths.append(original_width)

    return widths


def build_variant_path(source: Path, width: int) -> Path:
    relative_parent = source.parent.relative_to(IMAGES_DIR)
    output_dir = OPTIMIZED_DIR / relative_parent
    return output_dir / f"{source.stem}-{width}.webp"


def optimize_variant(source: Path, destination: Path, width: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "magick",
        str(source),
        "-auto-orient",
        "-strip",
        "-resize",
        f"{width}x>",
        "-quality",
        "82",
        "-define",
        "webp:method=6",
        "-define",
        "webp:auto-filter=true",
        "-define",
        "webp:target-size=0",
        str(destination),
    ]

    run_command(command)


def variant_is_current(source: Path, destination: Path) -> bool:
    return destination.exists() and destination.stat().st_mtime_ns >= source_mtime_ns(
        source
    )


def identify_variant_with_repair(
    source: Path, destination: Path, width: int
) -> tuple[int, int, bool]:
    rebuilt = False

    try:
        variant_width, variant_height = identify_image(destination)
    except RuntimeError:
        optimize_variant(source, destination, width)
        variant_width, variant_height = identify_image(destination)
        rebuilt = True

    return variant_width, variant_height, rebuilt


def public_path(path: Path) -> str:
    return "/" + path.relative_to(ROOT_DIR).as_posix()


def build_variant_index(variants: object) -> dict[int, dict[str, object]]:
    if not isinstance(variants, list):
        return {}

    variant_index = {}

    for variant in variants:
        if not isinstance(variant, dict):
            continue

        width = variant.get("width")

        if isinstance(width, int):
            variant_index[width] = variant

    return variant_index


def entry_is_current(source: Path, existing_entry: object) -> bool:
    if not isinstance(existing_entry, dict):
        return False

    if existing_entry.get("source_mtime_ns") != source_mtime_ns(source):
        return False

    extension = source.suffix.lower()

    if extension not in RASTER_EXTENSIONS:
        return True

    original_width = existing_entry.get("width")

    if not isinstance(original_width, int):
        return False

    variants = build_variant_index(existing_entry.get("variants"))

    for width in eligible_widths(original_width):
        variant = variants.get(width)

        if not variant:
            return False

        variant_path = variant.get("path")

        if not isinstance(variant_path, str):
            return False

        destination = ROOT_DIR / variant_path.lstrip("/")

        if not variant_is_current(source, destination):
            return False

    return True


def build_manifest_entry(
    source: Path, existing_entry: object
) -> tuple[str, dict[str, object], str] | None:
    extension = source.suffix.lower()
    source_key = public_path(source)
    current_mtime = source_mtime_ns(source)

    if entry_is_current(source, existing_entry):
        return source_key, existing_entry, "cached"

    if extension not in RASTER_EXTENSIONS:
        width = height = None

        try:
            width, height = identify_image(source)
        except RuntimeError:
            pass

        return (
            source_key,
            {
                "path": source_key,
                "format": extension.lstrip("."),
                "type": "original",
                "width": width,
                "height": height,
                "source_mtime_ns": current_mtime,
            },
            "indexed",
        )

    original_width, original_height = identify_image(source)
    variants = []
    optimized_any = False
    previous_variants = build_variant_index(
        existing_entry.get("variants") if isinstance(existing_entry, dict) else None
    )

    for width in eligible_widths(original_width):
        destination = build_variant_path(source, width)

        if not variant_is_current(source, destination):
            optimize_variant(source, destination, width)
            optimized_any = True

        cached_variant = previous_variants.get(width)

        if (
            isinstance(cached_variant, dict)
            and cached_variant.get("path") == public_path(destination)
            and isinstance(cached_variant.get("height"), int)
            and isinstance(cached_variant.get("width"), int)
            and variant_is_current(source, destination)
        ):
            variant_width, variant_height, rebuilt = identify_variant_with_repair(
                source, destination, width
            )
            optimized_any = optimized_any or rebuilt
        else:
            variant_width, variant_height, rebuilt = identify_variant_with_repair(
                source, destination, width
            )
            optimized_any = optimized_any or rebuilt

        variants.append(
            {
                "path": public_path(destination),
                "width": variant_width,
                "height": variant_height,
                "format": "webp",
            }
        )

    return (
        source_key,
        {
            "path": source_key,
            "format": extension.lstrip("."),
            "type": "raster",
            "width": original_width,
            "height": original_height,
            "source_mtime_ns": current_mtime,
            "variants": variants,
        },
        "optimized" if optimized_any else "cached",
    )


def iter_source_images() -> list[Path]:
    files = []

    for path in IMAGES_DIR.rglob("*"):
        if not path.is_file():
            continue

        if any(part in SKIP_DIRS for part in path.relative_to(IMAGES_DIR).parts):
            continue

        files.append(path)

    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responsive WebP variants and a manifest for site images."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional image paths relative to the repository root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of images to process in parallel (default: {DEFAULT_WORKERS}).",
    )
    return parser.parse_args()


def load_existing_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {}

    return json.loads(MANIFEST_PATH.read_text())


def resolve_sources(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return iter_source_images()

    resolved = []

    for raw_path in raw_paths:
        candidate = (ROOT_DIR / raw_path).resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {raw_path}")

        if IMAGES_DIR not in candidate.parents and candidate != IMAGES_DIR:
            raise ValueError(f"Image must be inside {IMAGES_DIR}: {raw_path}")

        resolved.append(candidate)

    return sorted(resolved)


def main() -> int:
    args = parse_args()

    try:
        sources = resolve_sources(args.paths)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 1

    manifest: dict[str, object] = load_existing_manifest()
    failure_detected = False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                build_manifest_entry, source, manifest.get(public_path(source))
            ): source
            for source in sources
        }

        for future in as_completed(future_map):
            source = future_map[future]

            try:
                entry = future.result()
            except Exception as error:
                print(f"failed {public_path(source)}: {error}", file=sys.stderr)
                failure_detected = True
                continue

            if entry is None:
                continue

            key, value, status = entry
            manifest[key] = value
            print(f"{status} {key}")

    if failure_detected:
        return 1

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
