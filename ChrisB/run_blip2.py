# run_blip2.py

import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
import os

# Parse arguments
parser = argparse.ArgumentParser(description="BLIP-2 protein image captioning")
parser.add_argument("--image", type=str, required=True, help="Path to the image file")
args = parser.parse_args()

image_path = args.image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# Open and preprocess image
raw_image = Image.open(image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate caption
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
