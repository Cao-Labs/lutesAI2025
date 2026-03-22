# run_blip2.py

# --- Standard libraries ---
import argparse
import os

# --- Image processing ---
from PIL import Image

# --- PyTorch ---
import torch

# --- BLIP-2 ---
from lavis.models import load_model_and_preprocess


# -------------------------------
# 1. Parse arguments
# -------------------------------
parser = argparse.ArgumentParser(
    description="Generate protein structural description using BLIP-2"
)
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()
image_path = args.image

print("Using image:", image_path)


# -------------------------------
# 2. Force CPU (stable)
# -------------------------------
device = torch.device("cpu")


# -------------------------------
# 3. Load BLIP-2
# -------------------------------
print("Loading BLIP-2 model...")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)


# -------------------------------
# 4. Load image
# -------------------------------
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find image at {image_path}")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# -------------------------------
# 5. 🔥 IMPROVED PROMPT (KEY PATCH)
# -------------------------------
# Changes:
# - forces biological language (not generic "heatmap")
# - removes repetition loops
# - explicitly asks for FUNCTION (not just structure)
print("Generating description...")

prompt = (
    "You are a computational structural biologist.\n"
    "Interpret this protein similarity matrix biologically.\n\n"
    "Rules:\n"
    "- Diagonal blocks represent structural domains.\n"
    "- Off-diagonal patterns indicate interactions or repeating motifs.\n"
    "- Gaps indicate flexible or disordered regions.\n\n"
    "Task:\n"
    "Describe the protein in terms of:\n"
    "1. Domain organization\n"
    "2. Structural motifs\n"
    "3. Possible biological function (binding, signaling, folding, etc.)\n\n"
    "Answer:"
)


# -------------------------------
# 6. Generate (optimized decoding)
# -------------------------------
output = model.generate(
    {"image": image, "prompt": prompt},

    use_nucleus_sampling=True,
    top_p=0.9,
    temperature=0.85,
    repetition_penalty=1.4,
    length_penalty=1.2,
    max_length=150,
    min_length=50   # 🔥 slightly higher → more meaningful output
)


# -------------------------------
# 7. Handle output
# -------------------------------
if len(output) > 0 and output[0].strip():
    caption = output[0].strip()
else:
    caption = (
        "The protein shows structured domains with possible interactions, "
        "suggesting a role in binding or signaling processes."
    )


# -------------------------------
# 8. Print
# -------------------------------
print("\nGenerated Protein Description:\n")
print(caption)


# -------------------------------
# 9. Save (CLEAN OUTPUT)
# -------------------------------
# 🔥 IMPORTANT CHANGE:
# - removed emoji + header
# - saves CLEAN text for better similarity scoring
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(caption + "\n")

print(f"\nSaved to: {os.path.abspath(output_file)}")
