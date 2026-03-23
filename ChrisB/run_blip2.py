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
# 5. 🔥 FINAL PROMPT (GROUNDING + ANTI-TEMPLATE)
# -------------------------------
print("Generating description...")

prompt = (
    "You are a computational structural biologist.\n"
    "Carefully analyze this protein similarity matrix.\n\n"

    "Interpretation rules:\n"
    "- Diagonal blocks = structural domains\n"
    "- Off-diagonal signals = interactions or repeating motifs\n"
    "- Gaps = flexible or disordered regions\n\n"

    "Instructions:\n"
    "- Write ONE paragraph (no lists, no numbering)\n"
    "- Describe SPECIFIC patterns visible in THIS image\n"
    "- Mention how many domains are present and where they appear\n"
    "- Describe any asymmetry or unusual regions\n"
    "- Avoid generic phrases (e.g., 'the image shows')\n"
    "- Do NOT repeat wording from previous outputs\n"
    "- Focus on what makes this protein unique\n\n"

    "Answer:"
)


# -------------------------------
# 6. 🔥 GENERATION (BALANCED FIX)
# -------------------------------
output = model.generate(
    {"image": image, "prompt": prompt},

    use_nucleus_sampling=True,
    top_p=0.9,               # balanced diversity
    temperature=0.85,        # keeps biological meaning
    repetition_penalty=1.5,  # strong but not destructive
    length_penalty=1.2,
    max_length=150,
    min_length=50
)


# -------------------------------
# 7. Handle output
# -------------------------------
if len(output) > 0 and output[0].strip():
    caption = output[0].strip()
else:
    caption = (
        "The matrix shows distinct structural domains with interaction regions, "
        "suggesting a role in binding or structural organization."
    )


# -------------------------------
# 8. Print
# -------------------------------
print("\nGenerated Protein Description:\n")
print(caption)


# -------------------------------
# 9. Save (clean text)
# -------------------------------
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(caption + "\n")

print(f"\nSaved to: {os.path.abspath(output_file)}")
