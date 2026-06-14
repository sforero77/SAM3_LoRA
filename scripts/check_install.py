#!/usr/bin/env python3
"""
Quick pre-flight check for the SAM3 LoRA pipeline.

Run this before kicking off a real training job. It verifies that all
core imports resolve, the prompt registry parses, the canonical config
template loads, and pycocotools can decode a sample polygon. Does NOT
load the SAM3 model itself (that requires GPU + HF cache and is heavy).

Usage:
    python scripts/check_install.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = True


def check(name: str, fn):
    global ok
    try:
        fn()
        print(f"✅ {name}")
    except Exception as e:
        ok = False
        print(f"❌ {name}: {e}")


def import_core():
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import yaml  # noqa: F401
    import pycocotools.mask  # noqa: F401


def import_pipeline():
    import lora_layers  # noqa: F401
    import inference_lora  # noqa: F401
    import train_sam3_lora_with_categories  # noqa: F401
    import train_class_lora  # noqa: F401
    import predict_class  # noqa: F401


def load_config_template():
    import yaml
    with open(ROOT / "configs" / "aerial_class_lora.yaml") as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "lora", "training", "output"):
        if key not in cfg:
            raise RuntimeError(f"missing top-level key '{key}' in template")


def load_prompt_registry():
    import yaml
    with open(ROOT / "prompts" / "class_prompts.yaml") as f:
        reg = yaml.safe_load(f)
    if not isinstance(reg, dict) or not reg:
        raise RuntimeError("prompt registry is empty or malformed")


def decode_sample_polygon():
    from pycocotools import mask as cm
    rles = cm.frPyObjects([[10, 10, 30, 10, 30, 30, 10, 30]], 50, 50)
    rle = cm.merge(rles)
    mask = cm.decode(rle)
    if mask.sum() == 0:
        raise RuntimeError("pycocotools decoded an empty mask for a 20x20 square")


def cuda_available():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False — training requires a CUDA GPU. "
            "There is no usable CPU training path (the SAM3 model + triton loss "
            "kernels need CUDA). Run on a Linux+CUDA host."
        )


print("🔎 Running pre-flight checks …\n")

check("core deps (torch / torchvision / numpy / PIL / yaml / pycocotools)", import_core)
check("pipeline modules import", import_pipeline)
check("configs/aerial_class_lora.yaml template loads", load_config_template)
check("prompts/class_prompts.yaml registry loads", load_prompt_registry)
check("pycocotools polygon decode works", decode_sample_polygon)
check("CUDA available", cuda_available)

print()
if ok:
    print("🎉 All checks passed. You can launch training.")
    sys.exit(0)
else:
    print("💥 At least one check failed. Fix the items above before training.")
    sys.exit(1)
