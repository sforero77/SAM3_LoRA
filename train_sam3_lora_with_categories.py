#!/usr/bin/env python3
"""
SAM3 LoRA Training with PROPER Category Support

This version:
- Reads category information from COCO file
- Uses actual class names as text prompts during training
- Supports multiple classes properly
"""

import os
import json
import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import numpy as np
from torchvision.transforms import v2
from tqdm import tqdm
from pycocotools import mask as coco_mask


# Geometric augmentations safe for nadir / overhead imagery.
# Each op is invariant under the aerial viewpoint (no horizon to break).
AUG_OPS = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")

DEFAULT_PROMPT_REGISTRY = Path(__file__).parent / "prompts" / "class_prompts.yaml"

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

from lora_layers import LoRAConfig, apply_lora_to_model, save_lora_weights, count_parameters


class SAM3DatasetWithCategories(Dataset):
    """Dataset that properly uses category names from COCO annotations"""

    def __init__(
        self,
        root_dir,
        coco_file_path=None,
        target_class: Optional[str] = None,
        augment: bool = False,
        prompt_registry_path: Optional[Path] = None,
        photometric_jitter: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        self.target_class = target_class
        self.augment = bool(augment)
        self.photometric_jitter = bool(photometric_jitter) and self.augment

        # Load the prompt synonym registry once. Missing file is non-fatal:
        # we just fall back to the COCO category name as the prompt.
        registry_path = prompt_registry_path or DEFAULT_PROMPT_REGISTRY
        self._prompt_registry = self._load_prompt_registry(registry_path)
        self._warned_missing_prompts: set = set()

        # Load COCO file to get category mappings
        if coco_file_path is None:
            coco_file_path = self.root_dir / "_annotations.coco.json"

        with open(coco_file_path, 'r') as f:
            self.coco_data = json.load(f)

        # Build category mapping: category_id -> category_name
        self.categories = {
            cat['id']: cat['name']
            for cat in self.coco_data['categories']
        }
        print(f"📂 Loaded {len(self.categories)} categories:")
        for cat_id, cat_name in self.categories.items():
            print(f"   - ID {cat_id}: '{cat_name}'")

        # Resolve target class (single-class filter)
        self._allowed_category_ids = None
        if target_class is not None:
            target_norm = target_class.strip().lower()
            matches = [
                cid for cid, cname in self.categories.items()
                if cname.strip().lower() == target_norm
            ]
            if not matches:
                available = sorted(self.categories.values())
                raise ValueError(
                    f"target_class='{target_class}' not found in COCO categories. "
                    f"Available: {available}"
                )
            self._allowed_category_ids = set(matches)
            self.target_category_id = matches[0]
            self.target_class_name = self.categories[matches[0]]
            print(
                f"🎯 Single-class filter active: '{self.target_class_name}' "
                f"(category_id={self.target_category_id})"
            )

        # Build mapping: image_filename -> list of (bbox, mask, category_id)
        # Apply single-class filter here if requested.
        self.image_annotations = {}
        total_anns = 0
        kept_anns = 0
        for ann in self.coco_data['annotations']:
            total_anns += 1
            image_id = ann['image_id']

            cat_id = ann.get('category_id', 1)
            if self._allowed_category_ids is not None and cat_id not in self._allowed_category_ids:
                continue

            # Find image filename
            image_info = next((img for img in self.coco_data['images'] if img['id'] == image_id), None)
            if not image_info:
                continue

            # Key by basename so a subdir-prefixed file_name in the COCO
            # (e.g. "images/tile_0001.png", common from some exporters) still
            # matches the on-disk tile, which is compared by p.name below.
            filename = os.path.basename(str(image_info['file_name']).replace("\\", "/"))

            if filename not in self.image_annotations:
                self.image_annotations[filename] = []

            self.image_annotations[filename].append({
                'bbox': ann['bbox'],  # [x, y, width, height] in COCO format
                'segmentation': ann.get('segmentation'),
                'category_id': cat_id,
                'area': ann.get('area', 0)
            })
            kept_anns += 1

        if self._allowed_category_ids is not None:
            print(
                f"🔎 Annotations after class filter: {kept_anns}/{total_anns} kept"
            )

        # Get image files; drop those without surviving annotations when filtering by class.
        # Discover tiles case-insensitively across every supported RGB extension
        # (the input contract advertises .jpg/.png/.tif). iterdir + suffix.lower()
        # avoids both the case-sensitivity gap on Linux and the double-counting a
        # list of per-extension globs would cause on case-insensitive filesystems.
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        all_image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        )
        if self._allowed_category_ids is not None:
            self.image_files = [
                p for p in all_image_files if p.name in self.image_annotations
            ]
            print(
                f"📷 Images after class filter: "
                f"{len(self.image_files)}/{len(all_image_files)} from {self.images_dir}"
            )
        else:
            self.image_files = all_image_files
            print(f"📷 Loaded {len(self.image_files)} images from {self.images_dir}")

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images left in {self.images_dir} after filtering. "
                f"Check target_class and your COCO annotations."
            )

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Modest photometric jitter applied only on the train split.
        self.color_jitter = v2.ColorJitter(brightness=0.15, contrast=0.15) \
            if self.photometric_jitter else None

        if self.augment:
            print("🔄 Augmentation enabled: hflip/vflip/rot90/rot180/rot270"
                  + (" + photometric jitter" if self.photometric_jitter else ""))

    @staticmethod
    def _load_prompt_registry(path: Path) -> Dict[str, Dict]:
        if not Path(path).is_file():
            return {}
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            # Normalise keys to lowercase for case-insensitive lookup.
            return {k.lower(): v for k, v in data.items() if isinstance(v, dict)}
        except Exception as e:
            print(f"⚠️  Could not read prompt registry at {path}: {e}")
            return {}

    def _sample_prompt(self, class_name: str) -> str:
        """
        Return a prompt string for the given canonical class name.
        - augment=True and class in registry → random.choice over synonyms + description.
        - augment=False and class in registry → first synonym (canonical).
        - Class not in registry → class_name as-is (with a one-time warning).
        """
        key = class_name.strip().lower()
        entry = self._prompt_registry.get(key)
        if entry is None:
            if key not in self._warned_missing_prompts:
                print(
                    f"ℹ️  Class '{class_name}' not in prompt registry — using "
                    f"the COCO name as-is. Add it to prompts/class_prompts.yaml "
                    f"to enable prompt augmentation."
                )
                self._warned_missing_prompts.add(key)
            return class_name

        synonyms = list(entry.get("synonyms") or [])
        if not synonyms:
            synonyms = [class_name]

        if self.augment:
            pool = list(synonyms) + list(entry.get("description") or [])
            return random.choice(pool)
        return synonyms[0]

    @staticmethod
    def _pick_aug_op() -> str:
        return random.choice(AUG_OPS)

    @staticmethod
    def _apply_aug_to_pil(img: PILImage.Image, op: str) -> PILImage.Image:
        if op == "identity":
            return img
        if op == "hflip":
            return img.transpose(PILImage.FLIP_LEFT_RIGHT)
        if op == "vflip":
            return img.transpose(PILImage.FLIP_TOP_BOTTOM)
        if op == "rot90":
            return img.transpose(PILImage.ROTATE_270)  # PIL ROTATE_270 == 90° clockwise
        if op == "rot180":
            return img.transpose(PILImage.ROTATE_180)
        if op == "rot270":
            return img.transpose(PILImage.ROTATE_90)   # PIL ROTATE_90 == 90° counter-clockwise
        raise ValueError(f"unknown aug op: {op}")

    @staticmethod
    def _apply_aug_to_bbox(
        bbox_xyxy: torch.Tensor, op: str, size: int
    ) -> torch.Tensor:
        """Transform a single [x1,y1,x2,y2] bbox by op, in a SIZE×SIZE image."""
        x1, y1, x2, y2 = bbox_xyxy.tolist()
        s = float(size)
        if op == "identity":
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        elif op == "hflip":
            nx1, ny1, nx2, ny2 = s - x2, y1, s - x1, y2
        elif op == "vflip":
            nx1, ny1, nx2, ny2 = x1, s - y2, x2, s - y1
        elif op == "rot90":   # clockwise 90: (x,y) -> (s-y, x)
            nx1, ny1, nx2, ny2 = s - y2, x1, s - y1, x2
        elif op == "rot180":  # (x,y) -> (s-x, s-y)
            nx1, ny1, nx2, ny2 = s - x2, s - y2, s - x1, s - y1
        elif op == "rot270":  # counter-clockwise 90: (x,y) -> (y, s-x)
            nx1, ny1, nx2, ny2 = y1, s - x2, y2, s - x1
        else:
            raise ValueError(f"unknown aug op: {op}")
        return torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float32)

    @staticmethod
    def _apply_aug_to_mask(mask: torch.Tensor, op: str) -> torch.Tensor:
        if mask is None or op == "identity":
            return mask
        if op == "hflip":
            return torch.flip(mask, dims=[-1])
        if op == "vflip":
            return torch.flip(mask, dims=[-2])
        # torch.rot90(k=1) rotates counter-clockwise; we want clockwise for rot90.
        if op == "rot90":
            return torch.rot90(mask, k=-1, dims=[-2, -1])
        if op == "rot180":
            return torch.rot90(mask, k=2, dims=[-2, -1])
        if op == "rot270":
            return torch.rot90(mask, k=1, dims=[-2, -1])
        raise ValueError(f"unknown aug op: {op}")

    def __len__(self):
        return len(self.image_files)

    def _decode_segmentation(self, segmentation, orig_h, orig_w, filename, ann_idx):
        """
        Decode a COCO segmentation field (polygon list, RLE dict, or compressed RLE)
        into a binary mask tensor at (self.resolution, self.resolution).

        Returns a float32 tensor of shape (H, W) with values in {0.0, 1.0},
        or None if decoding fails or segmentation is missing/empty. The collator
        tolerates None via the is_valid_segment flag, so failures do not crash
        training — they only disable mask loss for that object.
        """
        if segmentation is None:
            return None

        try:
            if isinstance(segmentation, list):
                if len(segmentation) == 0:
                    return None
                rles = coco_mask.frPyObjects(segmentation, orig_h, orig_w)
                rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
                mask = coco_mask.decode(rle)
            elif isinstance(segmentation, dict):
                rle = segmentation
                if isinstance(rle.get("counts"), list):
                    rle = coco_mask.frPyObjects(rle, orig_h, orig_w)
                mask = coco_mask.decode(rle)
            else:
                return None

            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.sum() == 0:
                return None

            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize(
                (self.resolution, self.resolution), PILImage.NEAREST
            )
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            return torch.from_numpy(mask_np)
        except Exception as e:
            print(
                f"⚠️  Segmentation decode failed for {filename} ann#{ann_idx}: {e}. "
                f"Falling back to bbox-only for this object."
            )
            return None

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Load image
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image to the model's working resolution
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Pick a single augmentation op for this sample so image, bboxes and
        # masks all transform consistently. Geometric op first, photometric
        # jitter (image only) second.
        aug_op = self._pick_aug_op() if self.augment else "identity"
        pil_image = self._apply_aug_to_pil(pil_image, aug_op)
        if self.color_jitter is not None:
            pil_image = self.color_jitter(pil_image)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations from COCO data
        filename = img_path.name
        annotations = self.image_annotations.get(filename, [])

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        objects = []
        category_ids = []

        for i, ann in enumerate(annotations):
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to [x1, y1, x2, y2] and scale to working resolution
            box_tensor = torch.tensor([
                x * scale_w,
                y * scale_h,
                (x + w) * scale_w,
                (y + h) * scale_h
            ], dtype=torch.float32)

            # Decode COCO segmentation (polygons or RLE) into a binary mask
            # at the model's working resolution. NEAREST preserves binary edges.
            segment = self._decode_segmentation(
                ann.get('segmentation'), orig_h, orig_w, filename, i
            )

            # Apply the same geometric op to bbox and mask
            box_tensor = self._apply_aug_to_bbox(box_tensor, aug_op, self.resolution)
            segment = self._apply_aug_to_mask(segment, aug_op)

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
                object_id=i,
                segment=segment
            )
            objects.append(obj)
            category_ids.append(ann['category_id'])

        # If no annotations, create dummy
        if not objects:
            objects = []
            category_ids = []

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by their category
        cat_id_to_object_ids = defaultdict(list)
        for obj, cat_id in zip(objects, category_ids):
            cat_id_to_object_ids[cat_id].append(obj.object_id)

        # Create one query per category. The prompt text is sampled from
        # the synonym registry (with augmentation on the train split) so
        # the LoRA learns the visual concept rather than memorising one
        # exact word.
        queries = []
        if len(cat_id_to_object_ids) > 0:
            for cat_id, obj_ids in cat_id_to_object_ids.items():
                class_name = self.categories.get(cat_id, "object")
                query_text = self._sample_prompt(class_name)
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=idx,
                        original_image_id=idx,
                        original_category_id=cat_id,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


class SAM3TrainerWithCategories:
    """Trainer that properly uses category names"""

    def __init__(self, config_path, target_class: Optional[str] = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Allow target_class to be specified inside the config too; CLI wins.
        if target_class is None:
            target_class = self.config.get("training", {}).get("target_class")
        self.target_class = target_class

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            # Training (not inference): builds the model's internal Hungarian
            # matcher so forward() can compute matching indices. With the
            # default eval_mode=True the matcher is None and forward() crashes.
            eval_mode=False,
        )

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )

        self.model = apply_lora_to_model(self.model, lora_config)

        # Print parameter count
        stats = count_parameters(self.model)
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"  Trainable percentage: {stats['trainable_percentage']:.2f}%")

        self.model.to(self.device)

        # Setup datasets with category support
        train_cfg = self.config["training"]
        train_path = Path(train_cfg["train_data_path"])

        # Mixed precision + gradient accumulation — honor the resolved config
        # (previously these keys were written but ignored, so training silently
        # ran fp32 with batch_size=1 regardless of what the config claimed).
        self.mixed_precision = str(train_cfg.get("mixed_precision", "no")).lower()
        self.grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

        print(f"\n📁 Loading datasets...")
        self.train_dataset = SAM3DatasetWithCategories(
            root_dir=train_path,
            coco_file_path=train_path / "_annotations.coco.json",
            target_class=self.target_class,
            augment=True,
        )

        # Validation dataset (no augmentation — deterministic)
        val_path = Path(train_cfg.get("val_data_path", "data/valid"))
        if val_path.exists() and (val_path / "_annotations.coco.json").exists():
            self.val_dataset = SAM3DatasetWithCategories(
                root_dir=val_path,
                coco_file_path=val_path / "_annotations.coco.json",
                target_class=self.target_class,
                augment=False,
            )
            print(f"✅ Validation data loaded: {len(self.val_dataset)} images")
        else:
            self.val_dataset = None
            print("⚠️ No validation data found")

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
            pin_memory=True
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=train_cfg.get("num_workers", 4),
                collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
                pin_memory=True
            )
        else:
            self.val_loader = None

        # Loss — Sam3LossWrapper with Hungarian matching, mirroring the proven
        # setup in validate_sam3_lora.py and the legacy native trainer. (The old
        # `SAM3Loss()` never existed in the vendored SAM3 code, so this trainer
        # could not even import before.)
        self._unwrapped_model = self.model
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )
        loss_fns = [
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_focal=False, alpha=0.25, gamma=2,
                use_presence=True, pad_n_queries=200,
            ),
            Masks(
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                focal_alpha=0.25, focal_gamma=2.0, compute_aux=False,
            ),
        ]
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",
            normalize_by_valid_object_num=False,
        )

        # Optimizer (only LoRA parameters)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=(train_cfg.get("adam_beta1", 0.9), train_cfg.get("adam_beta2", 0.999)),
            eps=train_cfg.get("adam_epsilon", 1e-8)
        )

        # LR Scheduler — one scheduler step per *optimizer* step (i.e. per
        # gradient-accumulation window), so T_max counts effective steps.
        steps_per_epoch = max(1, len(self.train_loader) // self.grad_accum_steps)
        total_steps = steps_per_epoch * train_cfg["num_epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )

        # Output directory
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = train_cfg["num_epochs"]
        self.best_val_loss = float('inf')

    def train(self):
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        print(f"📊 Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"📊 Validation samples: {len(self.val_dataset)}")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            if self.val_loader:
                val_loss = self.validate_epoch(epoch)
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = self.output_dir / "best_lora_weights.pt"
                    save_lora_weights(self.model, best_path)
                    print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}")

            # Save last model
            last_path = self.output_dir / "last_lora_weights.pt"
            save_lora_weights(self.model, last_path)

        # Copy last to best if no validation
        if not self.val_loader:
            import shutil
            shutil.copy(last_path, self.output_dir / "best_lora_weights.pt")
            print(f"ℹ️ No validation - copied last epoch as best")

        print(f"\n✅ Training complete! Weights saved to {self.output_dir}")

    def _compute_loss(self, outputs_list, input_batch):
        """Run Sam3LossWrapper and return the scalar core loss.

        The model injects Hungarian matcher indices into the outputs during its
        TRAINING-mode forward (it owns the matcher, set because we build with
        eval_mode=False). In EVAL mode (validation) that internal matching is
        skipped, so we inject the indices here when they are missing. The loss
        wrapper only reads out["indices"]. Mirrors validate_sam3_lora.py.
        """
        find_targets = [
            self._unwrapped_model.back_convert(t) for t in input_batch.find_targets
        ]
        for targets in find_targets:
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(self.device)

        # Inject matcher indices for any output that lacks them (the eval-mode
        # forward does not compute matching). No-op during training.
        with SAM3Output.iteration_mode(
            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as outputs_iter:
            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                for outputs in stage_outputs:
                    if "indices" not in outputs:
                        outputs["indices"] = self.matcher(outputs, stage_targets)
                    for aux_out in outputs.get("aux_outputs", []):
                        if "indices" not in aux_out:
                            aux_out["indices"] = self.matcher(aux_out, stage_targets)

        loss_dict = self.loss_wrapper(outputs_list, find_targets)
        return loss_dict[CORE_LOSS_KEY]

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # bf16 autocast needs no GradScaler (unlike fp16); enabled only on CUDA.
        use_amp = self.mixed_precision == "bf16" and self.device.type == "cuda"
        accum = self.grad_accum_steps
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        self.optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_batch = batch["input"]

            # Move to device
            input_batch = self._move_to_device(input_batch, self.device)

            # Forward under autocast (bf16); loss + Hungarian matching run in
            # fp32 outside autocast for numerical stability.
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs_list = self.model(input_batch)
            loss = self._compute_loss(outputs_list, input_batch)

            # Scale so accumulated grads over `accum` micro-batches equal the
            # average gradient of one effective batch (size batch_size × accum).
            (loss / accum).backward()

            # Step the optimizer once per accumulation window (and at epoch end).
            if (step + 1) % accum == 0 or (step + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.train_loader))

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0

        use_amp = self.mixed_precision == "bf16" and self.device.type == "cuda"
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_batch = batch["input"]
                input_batch = self._move_to_device(input_batch, self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs_list = self.model(input_batch)
                loss = self._compute_loss(outputs_list, input_batch)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _move_to_device(self, obj, device):
        """Recursively move nested structures to device"""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [self._move_to_device(x, device) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        elif hasattr(obj, "__dataclass_fields__"):
            for field in obj.__dataclass_fields__:
                val = getattr(obj, field)
                setattr(obj, field, self._move_to_device(val, device))
            return obj
        return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/aerial_class_lora.yaml")
    parser.add_argument(
        "--target_class",
        type=str,
        default=None,
        help="If set, train only on annotations whose COCO category name matches "
             "this string (case-insensitive). Overrides 'training.target_class' "
             "in the config.",
    )
    args = parser.parse_args()

    trainer = SAM3TrainerWithCategories(args.config, target_class=args.target_class)
    trainer.train()
