# run_blip2.py
import argparse
from pathlib import Path
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# --- Parse command-line argument ---
parser = argparse.ArgumentParser(description="Generate a caption from an image using BLIP-2")
parser.add_argument("--image", "-i", required=True, help="Path to the image file")
args = parser.parse_args()

image_path = Path(args.image).expanduser().resolve()
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device,
)

# --- Open and preprocess image ---
raw_image = Image.open(str(image_path)).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Generate caption ---
caption = model.generate({"image": image})[0]
print(f"\n🧬 Generated Caption for {image_path.name}:\n{caption}")
