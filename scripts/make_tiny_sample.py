#!/usr/bin/env python3
"""
Generate the tiny end-to-end smoke-test dataset under samples/tiny/.

Synthesises a self-contained, deterministic COCO dataset (no real imagery
required) so the verification recipe in AERIAL_LORA_GUIDE.md section 6 runs
out-of-the-box:

    samples/tiny/
      train/images/   5 RGB tiles + _annotations.coco.json
      valid/images/   2 RGB tiles + _annotations.coco.json

Each tile contains a few axis-aligned "building" rectangles on a textured
background; every building is written to COCO with a bbox AND a polygon
segmentation (the pipeline needs masks, not just boxes). One class: 'building'.

Usage:
    python scripts/make_tiny_sample.py
    python scripts/make_tiny_sample.py --tile_size 512 --out samples/tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAME = "building"


def _deterministic_buildings(tile_idx: int, size: int):
    """Return a deterministic list of (x, y, w, h) rectangles for a tile.

    Positions are a fixed function of the tile index, so the fixture is
    reproducible and committable (no RNG state to vary between runs).
    """
    rng = np.random.default_rng(seed=1000 + tile_idx)
    n = 1 + (tile_idx % 3)  # 1..3 buildings per tile
    rects = []
    for _ in range(n):
        w = int(rng.integers(size // 8, size // 4))
        h = int(rng.integers(size // 8, size // 4))
        x = int(rng.integers(5, size - w - 5))
        y = int(rng.integers(5, size - h - 5))
        rects.append((x, y, w, h))
    return rects


def _make_tile(tile_idx: int, size: int):
    """Build one RGB tile image and its building rectangles."""
    rng = np.random.default_rng(seed=tile_idx)
    # One greenish tone per tile (stand-in for vegetation/ground). A solid
    # background keeps the committed PNG tiny (compresses to a few KB) while
    # still giving clear contrast against the grey "buildings".
    base = rng.integers(70, 110, size=3)
    bg = np.empty((size, size, 3), dtype=np.uint8)
    bg[:] = base.astype(np.uint8)
    bg[..., 1] = min(255, int(base[1]) + 30)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img)

    rects = _deterministic_buildings(tile_idx, size)
    for (x, y, w, h) in rects:
        # Grey rooftop with a darker outline — a crude but clearly segmentable
        # "building" against the green background.
        draw.rectangle([x, y, x + w, y + h], fill=(180, 175, 170), outline=(90, 88, 85), width=3)
    return img, rects


def _rect_polygon(x, y, w, h):
    return [float(x), float(y), float(x + w), float(y),
            float(x + w), float(y + h), float(x), float(y + h)]


def build_split(split_dir: Path, n_tiles: int, start_idx: int, size: int):
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "categories": [{"id": 1, "name": CLASS_NAME}],
        "annotations": [],
    }
    ann_id = 1
    for i in range(n_tiles):
        tile_idx = start_idx + i
        img, rects = _make_tile(tile_idx, size)
        file_name = f"tile_{tile_idx:04d}.png"
        img.save(images_dir / file_name)

        image_id = tile_idx
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": size,
            "height": size,
        })
        for (x, y, w, h) in rects:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [_rect_polygon(x, y, w, h)],
            })
            ann_id += 1

    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[ok] {split_dir}: {n_tiles} tiles, {ann_id - 1} annotations")


def main():
    parser = argparse.ArgumentParser(description="Generate the tiny smoke-test dataset.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "samples" / "tiny")
    parser.add_argument("--tile_size", type=int, default=512)
    args = parser.parse_args()

    build_split(args.out / "train", n_tiles=5, start_idx=1, size=args.tile_size)
    build_split(args.out / "valid", n_tiles=2, start_idx=101, size=args.tile_size)
    print(f"\nTiny sample written to {args.out}. Now run the section-6 smoke test.")


if __name__ == "__main__":
    main()
