# SAM3 LoRA — per-class specialist for aerial / satellite imagery

> One method, any class. Hand in a COCO dataset of class X → train a LoRA
> specialist for class X → predict masks of class X on new tiles.

This is the canonical entry point for the repository. Everything else under
`legacy/` is kept for reference only.

---

## What this gives you

- A reproducible recipe that turns a COCO dataset into a small LoRA file
  (~20–40 MB) specialised for a single class.
- Works on any class your dataset contains: `building`, `vegetation`,
  `avocado_tree`, `lemon_tree`, `road`, `water`, `greenhouse`, etc. The
  granularity is decided by the labels in your COCO.
- One LoRA per class. Mix and match at inference time.

---

## 1. Input contract (what you must provide)

The pipeline expects RGB tiles + a COCO annotation file per split. You
prepare tiling and georeferencing **before** the pipeline; nothing in
this guide handles GeoTIFFs, multispectral bands, or world coordinates.

### Directory layout

```
<dataset_root>/
  train/
    images/                      # *.jpg | *.png | *.tif (RGB)
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
```

- Tiles are resized internally to 1008×1008. 1024×1024 source tiles work
  best (small downscale).
- RGB only. Strip alpha channels and extra bands before this stage.

### COCO format (required fields)

- `images[]`: `id`, `file_name` (must match files inside `images/`), `width`, `height`.
- `categories[]`: `id`, `name`. The **name** is the text prompt for SAM3 —
  pick concise names like `building`, `vegetation`, `avocado_tree`. Match
  the entry you'll add to `prompts/class_prompts.yaml` (optional).
- `annotations[]`:
  - `image_id`, `category_id` — required.
  - `bbox: [x, y, w, h]` — required, original-image pixels.
  - `segmentation` — **required**. Either a polygon list `[[x1,y1,x2,y2,...]]`
    or a COCO RLE dict `{"counts":..., "size":[h,w]}`. Without this the
    LoRA can only learn bbox proposals, which is useless for vegetation /
    buildings where boundaries matter.
  - `area` (recommended), `iscrowd` (optional, default 0).

A multi-class COCO is fine; the pipeline keeps only the annotations whose
category matches `--target_class`. You can re-use the same COCO file to
train several class-specific LoRAs.

### Minimal example

```json
{
  "images": [
    {"id": 1, "file_name": "tile_0001.png", "width": 1024, "height": 1024}
  ],
  "categories": [
    {"id": 1, "name": "avocado_tree"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [120, 340, 80, 60], "area": 4800, "iscrowd": 0,
     "segmentation": [[120,340, 200,340, 200,400, 120,400]]}
  ]
}
```

### Dataset sanity

| Check | Recommended |
|---|---|
| Train tiles for the target class | ≥ 200 |
| Valid tiles for the target class | ≥ 50 |
| Fraction of empty tiles in a split | < 30 % |
| Image format | JPG / PNG, RGB |
| Tile dimensions | 1024×1024 (any square works) |

---

## 2. Prompt registry (optional but recommended)

`prompts/class_prompts.yaml` maps a canonical class name to synonyms and
visual descriptors. During training, the prompt sent to SAM3 is sampled
randomly from this pool — the LoRA learns the visual concept, not a single
word, and stays robust to paraphrases / multilingual prompts at inference
time.

```yaml
avocado_tree:
  parent: vegetation                          # documentation only
  synonyms: [avocado tree, árbol de aguacate, aguacate]
  description: [dense round canopy, dark green tree]
```

- If the class is not in the registry, the COCO name is used verbatim
  (with a one-time warning). Adding an entry is optional but improves
  robustness — especially for unusual classes whose pretrained CLIP-text
  embedding is weak.
- For nadir classes with multilingual deployment, list synonyms in every
  language you intend to query with.

---

## 3. Train

One command per class:

```bash
python train_class_lora.py \
  --dataset_root data/avocado \
  --target_class avocado_tree \
  --output_dir outputs/avocado_lora
```

Optional overrides: `--epochs`, `--batch_size`, `--learning_rate`,
`--config <path>` (defaults to `configs/aerial_class_lora.yaml`).

What happens:

1. The dataset contract is validated up-front (clear error if the layout
   is wrong or the class name isn't in `categories[]`).
2. `configs/aerial_class_lora.yaml` is rendered with `{{TARGET_CLASS}}`
   and `{{DATASET_ROOT}}` substituted, then persisted to
   `<output_dir>/resolved_config.yaml` so inference can re-load it.
3. The dataset is filtered to your target class (other categories
   dropped, images with no remaining annotations removed).
4. Geometric augmentations (`identity / hflip / vflip / rot90 / rot180 /
   rot270`) plus modest photometric jitter run on the train split.
   Validation stays deterministic.
5. Prompts are sampled from `prompts/class_prompts.yaml` (with
   augmentation on the train split).
6. The LoRA-only weights are saved to
   `<output_dir>/best_lora_weights.pt` (best val loss) and
   `<output_dir>/last_lora_weights.pt`.

Defaults in `configs/aerial_class_lora.yaml`: LoRA rank 16, alpha 32,
LoRA active on every component (vision, text, geometry, DETR encoder /
decoder, mask decoder), lr 1e-4, batch 1 × grad-accum 4 = effective 4,
bf16, cosine schedule, 40 epochs.

---

## 4. Predict

```bash
python predict_class.py \
  --lora_weights outputs/avocado_lora/best_lora_weights.pt \
  --class_name avocado_tree \
  --input_dir data/test_tiles \
  --output_dir outputs/avocado_predictions
```

For each input tile this writes `<output_dir>/masks/<tile_stem>_mask.png`:
a single-channel binary PNG at the **original tile resolution** with the
union of all detections above the score threshold (after NMS).

Useful flags:

- `--threshold 0.5` — score cutoff. Lower → more recall, more false positives.
- `--nms_iou 0.5` — NMS IoU. Lower → fewer overlapping detections.
- `--config <path>` — override config. Defaults to
  `<lora_weights>/../resolved_config.yaml`, which `train_class_lora.py`
  always writes.
- `--coco_output` — also emit `<output_dir>/predictions.coco.json` with
  bboxes, scores, and a rectangle-fallback segmentation.

To stitch predictions back into a geo-referenced raster, do it
**outside** this pipeline: the binary PNGs are at the original tile
resolution, so you can re-apply the affine transform of each source
tile and mosaic the results in your GIS of choice.

---

## 5. Add a new class

Three steps. No code changes needed.

1. **Update the registry** — append to `prompts/class_prompts.yaml`:

   ```yaml
   greenhouse:
     synonyms: [greenhouse, invernadero, plastic greenhouse]
     description: [transparent rectangular structure, agricultural plastic cover]
   ```

2. **Prepare a COCO** under `<dataset_root>/train` and `<dataset_root>/valid`
   whose `categories[].name` includes `"greenhouse"` (and optionally other
   classes — the filter will drop them).

3. **Train**:

   ```bash
   python train_class_lora.py \
     --dataset_root data/greenhouse \
     --target_class greenhouse \
     --output_dir outputs/greenhouse_lora
   ```

---

## 6. Verification recipe (tiny smoke test)

Use this to confirm the pipeline runs end-to-end before throwing real
data at it.

1. Populate `samples/tiny/train/images/` (5 tiles) and
   `samples/tiny/valid/images/` (2 tiles), all 1024×1024 RGB.
2. Hand-label each tile with a single class (e.g. `building`) and write
   `samples/tiny/train/_annotations.coco.json` and
   `samples/tiny/valid/_annotations.coco.json` following the contract
   above. Use polygons, not just bboxes.
3. Train for 3 epochs:

   ```bash
   python train_class_lora.py \
     --dataset_root samples/tiny \
     --target_class building \
     --output_dir outputs/tiny_test \
     --epochs 3
   ```

   Pass criteria: completes in < 5 min on a single GPU;
   `outputs/tiny_test/best_lora_weights.pt` exists and is < 50 MB.

4. Predict:

   ```bash
   python predict_class.py \
     --lora_weights outputs/tiny_test/best_lora_weights.pt \
     --class_name building \
     --input_dir samples/tiny/valid/images \
     --output_dir outputs/tiny_test/preds
   ```

   Pass criteria: at least one mask PNG is not all-zero.

5. Smoke-test the class filter: prepare a multi-class COCO and train with
   `--target_class building`; the log should show
   `N images → M after filtering to class 'building'`.

---

## 7. Project layout (active surface only)

| Path | Purpose |
|---|---|
| `train_class_lora.py` | Canonical CLI wrapper — one command per class. |
| `train_sam3_lora_with_categories.py` | Underlying trainer (dataset + loop). |
| `predict_class.py` | Canonical batch inference for a class LoRA. |
| `inference_lora.py` | Lower-level inference (used by `predict_class.py`). |
| `lora_layers.py` | LoRA injection into SAM3 components. |
| `validate_sam3_lora.py` | cgF1 / mAP evaluation. |
| `configs/aerial_class_lora.yaml` | Parameterised template. |
| `prompts/class_prompts.yaml` | Extensible class → synonyms registry. |
| `sam3/` | SAM3 model code (Facebook). |
| `sam3_lora/` | Project-specific LoRA helpers and dataset. |
| `legacy/` | Earlier scripts, configs and docs — not maintained. |

---

## 8. Troubleshooting

- **`target_class='X' not found in COCO categories`** — the name doesn't
  match any `categories[].name` (case-insensitive). Check the COCO file.
- **`No images left in <split>/images after filtering`** — after dropping
  other classes the split is empty. Add tiles that actually contain the
  target class.
- **`Segmentation decode failed for ...`** — pycocotools could not parse
  that `segmentation` field. The pipeline logs the file/annotation index,
  falls back to bbox-only for that one object, and keeps training.
- **Predictions look like just bounding boxes** — your COCO is missing
  the `segmentation` field on most annotations. The LoRA never saw masks
  during training. Re-label with polygons or RLE.
- **Class not in prompt registry warning** — non-fatal. The COCO name is
  used as-is. Add an entry to `prompts/class_prompts.yaml` if you want
  prompt augmentation and multilingual robustness.
