#!/usr/bin/env python3
"""Download the Caltech-101 dataset and copy all images into data/label_0."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchvision.datasets import Caltech101


def find_categories_dir(root: Path) -> Path:
    """Return the directory that contains the 101_ObjectCategories folder."""
    candidates = sorted(root.rglob("101_ObjectCategories"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate '101_ObjectCategories' within {root}."
            " Make sure the Caltech-101 download completed successfully."
        )
    return candidates[0]


def copy_images(source_root: Path, dest_dir: Path, limit: int | None = None) -> int:
    """Copy jpg images from source_root into dest_dir.

    Args:
        source_root: Directory that contains class sub-folders.
        dest_dir: Output directory where flattened images will be stored.
        limit: Optional upper bound on the number of images to copy.

    Returns:
        The number of images copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    class_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        for img_path in sorted(class_dir.glob("*.jpg")):
            dest_name = f"{class_dir.name}_{img_path.name}"
            dest_path = dest_dir / dest_name

            if dest_path.exists():
                continue  # Skip duplicates when re-running the script

            shutil.copy2(img_path, dest_path)
            copied += 1

            if limit is not None and copied >= limit:
                return copied

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/caltech101"),
        help="Download root directory for the Caltech-101 archive.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/label_0"),
        help="Destination directory for flattened photo images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images to copy (useful for quick tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Downloading Caltech-101 (if not present)...")
    Caltech101(root=str(args.root), download=True)

    categories_dir = find_categories_dir(args.root)
    print(f"Found categories directory at: {categories_dir}")

    copied = copy_images(categories_dir, args.dest, args.limit)
    print(f"Copied {copied} images into {args.dest}")


if __name__ == "__main__":
    main()
