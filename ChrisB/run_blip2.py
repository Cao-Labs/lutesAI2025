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
image_path = args.image

# --- Debug: confirm which file is being processed ---
print("Using image:", image_path)

# --- Choose device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load BLIP-2 model ---
print("Loading BLIP-2 model...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5base"
    is_eval=True,
    device=device
)

# --- Load and preprocess image ---
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find image at {image_path}. Check the path!")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Generate caption (FIX: add prompt) ---
print("Generating description...")

prompt = (
    "This image represents a protein similarity matrix derived from sequence embeddings. "
    "Describe possible structural features, domain organization, or biological function."
)

caption = model.generate({
    "image": image,
    "prompt": prompt
})[0]

print("\n🧬 Generated Protein Function Description:\n")
print(caption)

# --- Save output to a text file ---
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("🧬 Generated Protein Function Description:\n")
    f.write(caption + "\n")

print(f"\n✅ Description saved to: {output_file}")
