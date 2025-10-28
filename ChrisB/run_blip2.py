# run_blip2.py (debug-ready)

from PIL import Image
import torch
import argparse
from lavis.models import load_model_and_preprocess
import os
import sys

# -----------------------------
# Debug: which file and path is running
# -----------------------------
print("DEBUG: running file:", os.path.abspath(__file__))
print("DEBUG: sys.path =", sys.path)

# -----------------------------
# Command-line argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Generate protein image captions using BLIP-2")
parser.add_argument("--image", type=str, required=True, help="Path to input image")
args = parser.parse_args()
image_path = args.image

# -----------------------------
# DEBUG: Check which image path is being used
# -----------------------------
print("DEBUG: image_path =", image_path)

# -----------------------------
# Check if file exists
# -----------------------------
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file not found at: {image_path}")

# -----------------------------
# Choose device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEBUG: using device =", device)

# -----------------------------
# Load BLIP-2 model
# -----------------------------
print("DEBUG: Loading BLIP-2 model...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
print("DEBUG: Model loaded successfully.")

# -----------------------------
# Load and preprocess image
# -----------------------------
raw_image = Image.open(image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
print("DEBUG: Image loaded and preprocessed.")

# -----------------------------
# Generate caption
# -----------------------------
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
