import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to input image")
args = parser.parse_args()
image_path = args.image  # <-- use the CLI argument

# -----------------------------
# Load image
# -----------------------------
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: file not found: {image_path}")
    exit(1)

print(f"Loaded image: {image_path}")

# -----------------------------
# Load BLIP-2 model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",  # or "blip2_caption" if using that variant
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device
)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# -----------------------------
# Generate caption
# -----------------------------
with torch.no_grad():
    caption = model.generate({"image": image})

print("Generated caption:", caption[0])
