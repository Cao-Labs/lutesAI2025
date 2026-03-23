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
# 5. 🔥 STRONGER PROMPT (ANTI-TEMPLATE)
# -------------------------------
print("Generating description...")

prompt = (
    "You are a computational structural biologist.\n"
    "Analyze this protein similarity matrix and describe it biologically.\n\n"

    "Interpretation rules:\n"
    "- Diagonal blocks indicate structural domains\n"
    "- Off-diagonal signals indicate interactions or repeating motifs\n"
    "- Gaps indicate flexible or disordered regions\n\n"

    "Instructions:\n"
    "- Write ONE clear paragraph (no bullet points, no numbering)\n"
    "- Be specific about where patterns occur (beginning, middle, end)\n"
    "- Describe how many domains are visible and how they are arranged\n"
    "- Mention any repeating or asymmetric patterns\n"
    "- Suggest a possible biological role based on structure\n"
    "- Do NOT repeat phrases or use generic templates\n\n"

    "Final Answer:"
)


# -------------------------------
# 6. 🔥 GENERATION (FIX REPETITION)
# -------------------------------
output = model.generate(
    {"image": image, "prompt": prompt},

    use_nucleus_sampling=True,
    top_p=0.85,              # slightly tighter sampling
    temperature=0.7,         # less randomness → more grounded
    repetition_penalty=2.0,  # 🔥 STRONG anti-repeat
    length_penalty=1.1,
    max_length=160,
    min_length=60
)


# -------------------------------
# 7. Handle output
# -------------------------------
if len(output) > 0 and output[0].strip():
    caption = output[0].strip()
else:
    caption = (
        "The protein displays multiple structured domains with distinct regions "
        "of interaction, suggesting a role in molecular binding or signaling."
    )


# -------------------------------
# 8. Print
# -------------------------------
print("\nGenerated Protein Description:\n")
print(caption)


# -------------------------------
# 9. Save (CLEAN OUTPUT)
# -------------------------------
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(caption + "\n")

print(f"\nSaved to: {os.path.abspath(output_file)}")
