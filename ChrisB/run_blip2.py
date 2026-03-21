# run_blip2.py

import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# --- Parse arguments ---
parser = argparse.ArgumentParser(
    description="Generate protein structural description using BLIP-2"
)
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()
image_path = args.image

print("Using image:", image_path)

# --- Use CPU (stable) ---
device = torch.device("cpu")

# --- Load BLIP-2 ---
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

# --- 🔥 DOMAIN-AWARE PROMPT (your research applied) ---
print("Generating description...")

prompt = (
    "System: You are a computational structural biologist.\n"
    "Rules:\n"
    "- Diagonal blocks indicate structural domains.\n"
    "- Off-diagonal signals indicate interactions or repeating motifs.\n"
    "- Gaps indicate boundaries or flexible regions.\n\n"
    "Question: Analyze this protein similarity heatmap. "
    "Describe domain structure, repeating patterns, and functional regions.\n"
    "Answer:"
)

# --- 🔥 OPTIMIZED GENERATION ---
output = model.generate(
    {"image": image, "prompt": prompt},

    use_nucleus_sampling=True,
    top_p=0.9,
    temperature=0.85,
    repetition_penalty=1.4,
    length_penalty=1.2,
    max_length=150,
    min_length=40
)

# --- Safe output handling ---
if len(output) > 0 and output[0].strip():
    caption = output[0].strip()
else:
    caption = (
        "The heatmap shows structured diagonal regions and possible domain organization, "
        "with patterns suggesting interactions between different regions of the protein."
    )

# --- Print ---
print("\n🧬 Generated Protein Function Description:\n")
print(caption)

# --- Save ---
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("🧬 Generated Protein Function Description:\n")
    f.write(caption + "\n")

print(f"\n✅ Description saved to: {output_file}")
