# run_blip2.py

import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(
    description="Generate a protein function description using BLIP-2."
)
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to the input protein image"
)
args = parser.parse_args()
image_path = args.image  # <-- this ensures the command line input is used

# --- Debug: confirm which file is being processed ---
print("Using image:", image_path)

# --- Choose device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load BLIP-2 model (pretrained FLAN-T5 variant) ---
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# --- Load and preprocess image ---
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find image at {image_path}. Check the path!")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Generate caption ---
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
