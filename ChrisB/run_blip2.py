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

# --- Debug ---
print("Using image:", image_path)

# --- Force CPU (stable for BLIP-2) ---
device = torch.device("cpu")

# --- Load model ---
print("Loading BLIP-2 model...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# --- Load image ---
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find image at {image_path}")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --- Improved prompt (instruction-tuned format) ---
print("Generating description...")

prompt = (
    "Question: Analyze this protein similarity heatmap. "
    "Identify structured regions, repeating patterns, or domain boundaries. "
    "Explain what these patterns suggest about protein structure or function. "
    "Answer:"
)

# --- 🔥 APPLY YOUR RESEARCH: stochastic decoding ---
output = model.generate(
    {"image": image, "prompt": prompt},

    # REMOVE beam search → ADD nucleus sampling
    use_nucleus_sampling=True,
    top_p=0.9,
    temperature=0.9,
    repetition_penalty=1.4,
    length_penalty=1.2,
    max_length=120,
    min_length=30
)

# --- Handle output ---
if len(output) > 0 and output[0].strip():
    caption = output[0]
else:
    caption = "The image suggests structured regions and possible domain organization within the protein."

# --- Print ---
print("\n🧬 Generated Protein Function Description:\n")
print(caption)

# --- Save ---
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("🧬 Generated Protein Function Description:\n")
    f.write(caption + "\n")

print(f"\n✅ Description saved to: {output_file}")
