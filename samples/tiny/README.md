# Tiny verification sample

Slot for the end-to-end smoke test described in section 6 of
`AERIAL_LORA_GUIDE.md`.

## Expected layout

```
samples/tiny/
  train/
    images/                       # 5 RGB tiles, 1024x1024 recommended
    _annotations.coco.json
  valid/
    images/                       # 2 RGB tiles, same shape as train
    _annotations.coco.json
```

Both COCO files must follow the input contract from the guide:
`images[]`, `categories[]` (with at least one entry, e.g. `building`),
`annotations[]` with `bbox`, `category_id`, and a non-empty
`segmentation` field (polygon list or RLE). Polygons are easier to
hand-label quickly with tools like Roboflow / CVAT / LabelMe.

## Generate the fixture (no real data needed)

You don't have to hand-label anything: a generator synthesises a valid
`building` dataset (5 train + 2 valid RGB tiles with polygon masks) into
this directory:

```bash
python scripts/make_tiny_sample.py
```

This writes `train/` and `valid/` matching the layout above so the
section-6 smoke test below runs as-is.

## Run the smoke test

```bash
python train_class_lora.py \
  --dataset_root samples/tiny \
  --target_class building \
  --output_dir outputs/tiny_test \
  --epochs 3

python predict_class.py \
  --lora_weights outputs/tiny_test/best_lora_weights.pt \
  --class_name building \
  --input_dir samples/tiny/valid/images \
  --output_dir outputs/tiny_test/preds
```

Pass criteria:

- Training completes in < 5 minutes on a single GPU.
- `outputs/tiny_test/best_lora_weights.pt` exists and is < 50 MB.
- At least one prediction PNG in `outputs/tiny_test/preds/masks/` is
  not all-zero.

If the tiles are not already present, generate them first with
`python scripts/make_tiny_sample.py` (see above).
