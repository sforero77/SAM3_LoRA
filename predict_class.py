#!/usr/bin/env python3
"""
SAM3 LoRA — canonical batch inference for a per-class specialist.

Pairs with train_class_lora.py. Given a directory of tiles, it loads
the per-class LoRA weights, prompts SAM3 with the class name, applies
NMS, and saves one binary PNG mask per tile (union of all predicted
instances above the score threshold). Optionally also emits a
predictions.coco.json with bboxes/scores/segmentations.

Usage:
    python predict_class.py \\
        --lora_weights outputs/avocado_lora/best_lora_weights.pt \\
        --class_name avocado_tree \\
        --input_dir data/test_tiles \\
        --output_dir outputs/avocado_predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision.ops import nms

from inference_lora import SAM3LoRAInference


VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_PROMPT_REGISTRY = Path(__file__).parent / "prompts" / "class_prompts.yaml"


def _resolve_prompt(class_name: str, registry_path: Path) -> str:
    """Map a canonical class name to the prompt the LoRA was trained with.

    Training samples prompts from prompts/class_prompts.yaml and uses
    ``synonyms[0]`` as the canonical (non-augmented) prompt. Mirroring that here
    keeps the train-time and predict-time prompts consistent. Falls back to the
    raw class name when the registry or the entry is missing.
    """
    try:
        with open(registry_path, "r") as f:
            registry = yaml.safe_load(f) or {}
    except Exception:
        return class_name
    entry = registry.get(class_name.strip().lower())
    if not isinstance(entry, dict):
        return class_name
    synonyms = entry.get("synonyms") or []
    return synonyms[0] if synonyms else class_name


def collect_tiles(input_dir: Path) -> List[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def _post_process(
    predictions: dict,
    score_threshold: float,
    nms_iou: float,
):
    """
    Apply score threshold + NMS to the raw outputs of SAM3LoRAInference.predict
    and return:
      kept_indices, boxes_xyxy_orig (np.ndarray Nx4), scores (np.ndarray N), masks (np.ndarray NxHxW or None)
    All coordinates are in the ORIGINAL image pixel space.
    """
    boxes = predictions["boxes"]          # [N,4] in cxcywh normalised
    scores = predictions["scores"]        # [N, num_classes]
    masks = predictions["masks"]          # [N, h, w] or None
    orig_w, orig_h = predictions["original_size"]

    max_scores = scores.max(axis=1)
    valid = np.where(max_scores > score_threshold)[0]
    if len(valid) == 0:
        return valid, np.zeros((0, 4)), np.zeros((0,)), None

    cand_boxes = boxes[valid]
    cand_scores = max_scores[valid]
    cx, cy, w, h = cand_boxes[:, 0], cand_boxes[:, 1], cand_boxes[:, 2], cand_boxes[:, 3]
    x1 = np.clip((cx - w / 2) * orig_w, 0, orig_w)
    y1 = np.clip((cy - h / 2) * orig_h, 0, orig_h)
    x2 = np.clip((cx + w / 2) * orig_w, 0, orig_w)
    y2 = np.clip((cy + h / 2) * orig_h, 0, orig_h)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(
        torch.from_numpy(boxes_xyxy).float(),
        torch.from_numpy(cand_scores).float(),
        nms_iou,
    ).numpy()
    valid = valid[keep]
    boxes_xyxy = boxes_xyxy[keep]
    cand_scores = cand_scores[keep]

    out_masks = None
    if masks is not None:
        kept_masks = masks[valid]
        # Resize each mask to original image size with nearest-neighbour
        # (we threshold afterwards anyway).
        resized = np.zeros((len(kept_masks), orig_h, orig_w), dtype=np.float32)
        for i, m in enumerate(kept_masks):
            mp = PILImage.fromarray((m > 0).astype(np.uint8) * 255, mode="L")
            mp = mp.resize((orig_w, orig_h), PILImage.NEAREST)
            resized[i] = np.array(mp) > 127
        out_masks = resized

    return valid, boxes_xyxy, cand_scores, out_masks


def _coco_polygon_from_mask(mask: np.ndarray) -> List[List[float]]:
    """Cheap polygon approximation for the COCO export — just an outer rectangle.
    Real polygon contouring would need extra deps; the binary PNG carries the
    exact mask, this is a fallback for tools that only read segmentation."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [[float(x1), float(y1), float(x2), float(y1),
             float(x2), float(y2), float(x1), float(y2)]]


def run(args):
    weights_path = Path(args.lora_weights)
    if not weights_path.is_file():
        print(f"❌ LoRA weights not found: {weights_path}")
        sys.exit(2)

    # Resolve config: explicit flag wins, otherwise look for resolved_config.yaml
    # next to the weights (train_class_lora.py always writes one there).
    config_path = args.config
    if config_path is None:
        candidate = weights_path.parent / "resolved_config.yaml"
        if not candidate.is_file():
            print(
                f"❌ No --config given and {candidate} not found. "
                f"Either pass --config or train via train_class_lora.py "
                f"so a resolved_config.yaml is written next to the weights."
            )
            sys.exit(2)
        config_path = candidate
    config_path = Path(config_path)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    tiles = collect_tiles(input_dir)
    if not tiles:
        print(f"❌ No image tiles found in {input_dir} (extensions: {VALID_EXTS})")
        sys.exit(2)

    # Resolve the class prompt: prefer --class_name, else fall back to the
    # target_class persisted in the (resolved) config, then map it through the
    # prompt registry so it matches how training prompted the model.
    with open(config_path, "r") as f:
        cfg_for_prompt = yaml.safe_load(f) or {}
    class_name = args.class_name or cfg_for_prompt.get("training", {}).get("target_class")
    if not class_name:
        print("❌ No --class_name given and the config has no training.target_class. "
              "Pass --class_name explicitly.")
        sys.exit(2)
    prompt = _resolve_prompt(class_name, Path(args.prompts_yaml))

    print(f"🔧 Loading model with weights from {weights_path}")
    print(f"📝 Config: {config_path}")
    print(f"🎯 Class: {class_name!r}  →  SAM3 prompt: {prompt!r}")
    print(f"📁 Tiles: {len(tiles)} from {input_dir}")
    print(f"📦 Output: {output_dir}\n")

    inferencer = SAM3LoRAInference(str(config_path), str(weights_path))

    coco_images = []
    coco_annotations = []
    next_image_id = 1
    next_ann_id = 1
    total_detected = 0

    for tile_idx, tile_path in enumerate(tiles, start=1):
        print(f"[{tile_idx}/{len(tiles)}] {tile_path.name}")
        try:
            preds = inferencer.predict(str(tile_path), text_prompt=prompt)
        except Exception as e:
            print(f"  ⚠️  inference failed: {e}")
            continue

        kept, boxes_xyxy, kept_scores, kept_masks = _post_process(
            preds, args.threshold, args.nms_iou
        )

        orig_w, orig_h = preds["original_size"]
        union = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if kept_masks is not None and len(kept_masks) > 0:
            union = (kept_masks.sum(axis=0) > 0).astype(np.uint8) * 255
        elif len(boxes_xyxy) > 0:
            # No masks available, fall back to filled bboxes
            for x1, y1, x2, y2 in boxes_xyxy:
                union[int(y1):int(y2), int(x1):int(x2)] = 255

        out_mask_path = masks_dir / f"{tile_path.stem}_mask.png"
        PILImage.fromarray(union, mode="L").save(out_mask_path)
        total_detected += int(len(kept))
        print(f"  → {len(kept)} instance(s) above threshold {args.threshold}; "
              f"mask saved to {out_mask_path.relative_to(output_dir)}")

        if args.coco_output:
            image_id = next_image_id
            next_image_id += 1
            coco_images.append({
                "id": image_id,
                "file_name": tile_path.name,
                "width": int(orig_w),
                "height": int(orig_h),
            })
            for i in range(len(kept)):
                x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]
                seg = _coco_polygon_from_mask(kept_masks[i]) if kept_masks is not None else []
                coco_annotations.append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox_coco,
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                    "score": float(kept_scores[i]),
                    "segmentation": seg,
                })
                next_ann_id += 1

    if args.coco_output:
        coco = {
            "images": coco_images,
            "categories": [{"id": 1, "name": class_name}],
            "annotations": coco_annotations,
        }
        coco_path = output_dir / "predictions.coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"\n📝 COCO predictions written to {coco_path}")

    print(f"\n✅ Done. {total_detected} total detections across {len(tiles)} tiles.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for a SAM3 per-class LoRA specialist."
    )
    parser.add_argument(
        "--lora_weights",
        type=Path,
        required=True,
        help="Path to the trained LoRA .pt file.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="Canonical class the LoRA was trained on (e.g. 'avocado_tree'). "
             "Optional: if omitted, it is read from training.target_class in the "
             "resolved config. It is mapped through prompts/class_prompts.yaml so "
             "the inference prompt matches how training prompted the model.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory of tiles to segment. Tiles are processed independently.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write masks/ and (optionally) predictions.coco.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override path to the LoRA config YAML. Defaults to "
             "<lora_weights>/../resolved_config.yaml.",
    )
    parser.add_argument(
        "--prompts_yaml",
        type=Path,
        default=DEFAULT_PROMPT_REGISTRY,
        help="Prompt registry used to map the class name to its canonical "
             "synonym (default: prompts/class_prompts.yaml).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold (max-class-score) for keeping a prediction.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--coco_output",
        action="store_true",
        help="Also write predictions.coco.json with bboxes, scores and "
             "rectangle-segmentation fallback.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
