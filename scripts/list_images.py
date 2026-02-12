#!/usr/bin/env python3

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="List images with a given extension, output paths relative to a specified base root."
    )

    parser.add_argument(
        "--search-root",
        required=True,
        help="Directory to search in."
    )

    parser.add_argument(
        "--base-root",
        required=True,
        help="Base root directory to strip from resulting paths."
    )

    parser.add_argument(
        "--extension", "-e",
        required=True,
        help="File extension to match (e.g., jpg, png)."
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output text file to write relative paths into."
    )

    args = parser.parse_args()

    search_root = Path(args.search_root)
    base_root = Path(args.base_root)
    ext = args.extension.lower().lstrip(".")
    output_file = Path(args.output)

    results = []

    # Find matching files
    for file in search_root.rglob(f"*.{ext}"):
        try:
            relative_path = file.relative_to(base_root)
        except ValueError:
            # File is outside the base root; skip or warn
            continue

        results.append(str(relative_path))

    # Write to output file
    output_file.write_text("\n".join(results), encoding="utf-8")
    print(f"Found {len(results)} file(s). Paths written to {output_file}")

if __name__ == "__main__":
    main()

"""
python3 list_images.py \
    --base-root "/workspace/data/input/" \
    --extension png \
    --search-root "/workspace/data/input/openloris_location/" \
    --output ./train_
"""