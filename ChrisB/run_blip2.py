# run_blip2.py

import argparse
import os
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# --- Parse command-line argument ---
parser = argparse.ArgumentParser(description="Generate protein function caption using BLIP-2")
parser.add_argument("--image", type=str, required=True, help="Path to protein image (.png)")
args = parser.parse_args()

image_path = args.image

# --- Verify image exists ---
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

# --- Choose device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load BLIP-2 model (pretrained FLAN-T5 variant) ---
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# --- Load and preprocess protein image ---
raw_image = Image.open(image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Generate caption (natural-language description) ---
caption = model.generate({"image": image})[0]

print("\n🧬 Generated Protein Function Description:")
print(caption)
