#!/usr/bin/env python3
"""
Minimal BLIP-2 caption generator.
Usage:
  python run_blip2_min.py --image /full/path/to/image.png
"""

import sys
from pathlib import Path
import argparse

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Generate image caption using BLIP-2")
parser.add_argument("--image", "-i", type=str, required=True, help="Full path to the image file")
args = parser.parse_args()

image_path = Path(args.image).expanduser().resolve()
print(f"\n[INFO] Provided image path: {args.image}")
print(f"[INFO] Resolved path: {image_path}")
print(f"[INFO] Path exists: {image_path.exists()}")
if not image_path.exists():
    print("[ERROR] File not found! Check your path or mount points.")
    sys.exit(1)

# --- Heavy imports only after verifying file ---
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- Load model ---
print("[INFO] Loading BLIP-2 model (may print warnings)...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# --- Load image ---
try:
    raw_image = Image.open(str(image_path)).convert("RGB")
    print(f"[INFO] Successfully opened image, size: {raw_image.size}")
except Exception as e:
    print(f"[ERROR] Failed to open image: {e}")
    sys.exit(1)

# --- Preprocess and move to device ---
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Generate caption ---
caption = model.generate({"image": image})[0]
print("\n🧬 Generated Caption:\n", caption)
