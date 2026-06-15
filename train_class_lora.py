#!/usr/bin/env python3
"""
SAM3 LoRA — canonical per-class training entry point.

One LoRA per class. You hand in a COCO-formatted dataset (see
AERIAL_LORA_GUIDE.md for the input contract) and a target class
name; this wrapper resolves a config template, persists the
resolved config alongside the LoRA weights for later inference,
and runs the underlying trainer.

Example:
    python train_class_lora.py \\
        --dataset_root data/avocado \\
        --target_class avocado_tree \\
        --output_dir outputs/avocado_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from train_sam3_lora_with_categories import SAM3TrainerWithCategories


CONFIG_TEMPLATE_DEFAULT = "configs/aerial_class_lora.yaml"


def _substitute(value, mapping):
    if isinstance(value, str):
        for key, repl in mapping.items():
            value = value.replace("{{" + key + "}}", repl)
        return value
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    return value


def resolve_config(
    template_path: Path,
    dataset_root: Path,
    target_class: str,
    output_dir: Path,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
) -> dict:
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    mapping = {
        "TARGET_CLASS": target_class,
        "DATASET_ROOT": str(dataset_root),
    }
    cfg = _substitute(cfg, mapping)

    # Output dir always comes from --output_dir to keep the wrapper authoritative.
    cfg.setdefault("output", {})
    cfg["output"]["output_dir"] = str(output_dir)
    cfg["output"]["logging_dir"] = str(output_dir / "logs")

    train_cfg = cfg.setdefault("training", {})
    train_cfg["target_class"] = target_class
    if epochs is not None:
        train_cfg["num_epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size
    if learning_rate is not None:
        train_cfg["learning_rate"] = learning_rate

    return cfg


def validate_dataset(dataset_root: Path, target_class: str) -> None:
    """Fail fast with a clear message if the input contract is not met."""
    errors = []
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for split in ("train", "valid"):
        split_dir = dataset_root / split
        coco_path = split_dir / "_annotations.coco.json"
        images_dir = split_dir / "images"
        if not split_dir.is_dir():
            errors.append(f"missing directory: {split_dir}")
            continue
        if not coco_path.is_file():
            errors.append(f"missing COCO file: {coco_path}")
            continue
        # Images may be in an images/ subdir (contract) OR flat in the split dir
        # (Roboflow "COCO Segmentation" export). Accept either.
        if not images_dir.is_dir():
            has_flat_images = split_dir.is_dir() and any(
                p.is_file() and p.suffix.lower() in img_exts for p in split_dir.iterdir()
            )
            if not has_flat_images:
                errors.append(
                    f"no images found: expected {images_dir}/ or image files "
                    f"directly in {split_dir}"
                )
                continue
        try:
            with open(coco_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{coco_path}: invalid JSON ({e})")
            continue
        for key in ("images", "annotations", "categories"):
            if key not in data:
                errors.append(f"{coco_path}: missing '{key}' field")
        if "categories" in data:
            names = [c.get("name", "").lower() for c in data["categories"]]
            if target_class.lower() not in names:
                errors.append(
                    f"{coco_path}: target_class '{target_class}' not present "
                    f"in categories {sorted(set(names))}"
                )

    if errors:
        print("❌ Dataset contract violated:")
        for e in errors:
            print(f"   - {e}")
        print(
            "\nSee AERIAL_LORA_GUIDE.md for the required dataset layout and "
            "COCO fields."
        )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description="Train a per-class SAM3 LoRA specialist."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root directory containing train/ and valid/ subfolders, each "
             "with images/ and _annotations.coco.json.",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        required=True,
        help="Canonical class name. Must exactly match a categories[].name "
             "entry in the COCO file (case-insensitive).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write LoRA weights, logs and resolved_config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_TEMPLATE_DEFAULT),
        help=f"Config template (default: {CONFIG_TEMPLATE_DEFAULT}).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip the dataset contract pre-check (not recommended).",
    )

    args = parser.parse_args()

    if not args.skip_validation:
        validate_dataset(args.dataset_root, args.target_class)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(
        template_path=args.config,
        dataset_root=args.dataset_root,
        target_class=args.target_class,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    resolved_path = args.output_dir / "resolved_config.yaml"
    with open(resolved_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"📝 Resolved config written to {resolved_path}")
    print(f"🎯 Target class: {args.target_class}")
    print(f"📁 Dataset root: {args.dataset_root}")
    print(f"📦 Output dir:   {args.output_dir}\n")

    trainer = SAM3TrainerWithCategories(
        config_path=str(resolved_path),
        target_class=args.target_class,
    )
    trainer.train()


if __name__ == "__main__":
    main()
