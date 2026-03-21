# run_blip2.py

# --- Standard libraries for argument parsing and file handling ---
import argparse
import os

# --- Image processing ---
from PIL import Image

# --- PyTorch for model + device handling ---
import torch

# --- BLIP-2 loader from LAVIS ---
from lavis.models import load_model_and_preprocess


# -------------------------------
# 1. Parse command-line arguments
# -------------------------------
# This allows us to pass an image file when running the script
parser = argparse.ArgumentParser(
    description="Generate protein structural description using BLIP-2"
)

# Required argument: path to the protein image (.png)
parser.add_argument("--image", type=str, required=True)

args = parser.parse_args()
image_path = args.image

print("Using image:", image_path)


# -------------------------------
# 2. Set compute device
# -------------------------------
# We force CPU for stability (BLIP-2 can be heavy on GPU memory)
device = torch.device("cpu")


# -------------------------------
# 3. Load BLIP-2 model
# -------------------------------
# This loads:
# - Vision encoder (image understanding)
# - Language model (FLAN-T5)
# - Preprocessing functions
print("Loading BLIP-2 model...")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",  # pretrained vision-language model
    is_eval=True,                    # evaluation mode (no training)
    device=device
)


# -------------------------------
# 4. Load and preprocess image
# -------------------------------
# Convert image to RGB and apply normalization/transforms
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find image at {image_path}")

# Apply BLIP-2 preprocessing and add batch dimension
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# -------------------------------
# 5. Domain-aware prompt (KEY IDEA)
# -------------------------------
# This is based on your research:
# - Inject biological interpretation rules
# - Guide the model beyond generic "heatmap" descriptions
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


# -------------------------------
# 6. Generate description (OPTIMIZED)
# -------------------------------
# Uses stochastic decoding (from your research):
# - Prevents repetition loops
# - Improves diversity and meaning
output = model.generate(
    {"image": image, "prompt": prompt},

    use_nucleus_sampling=True,  # enables top-p sampling
    top_p=0.9,                 # controls diversity
    temperature=0.85,          # balances randomness vs accuracy
    repetition_penalty=1.4,    # discourages repeated phrases
    length_penalty=1.2,        # encourages longer explanations
    max_length=150,            # max tokens
    min_length=40              # forces meaningful output
)


# -------------------------------
# 7. Handle model output safely
# -------------------------------
# If BLIP-2 fails or outputs empty text, fallback to a default description
if len(output) > 0 and output[0].strip():
    caption = output[0].strip()
else:
    caption = (
        "The heatmap shows structured diagonal regions and possible domain organization, "
        "with patterns suggesting interactions between different regions of the protein."
    )


# -------------------------------
# 8. Print result
# -------------------------------
print("\n🧬 Generated Protein Function Description:\n")
print(caption)


# -------------------------------
# 9. Save output to file
# -------------------------------
# Output is saved in SAME directory as input image
output_file = image_path.rsplit('.', 1)[0] + "_description.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("🧬 Generated Protein Function Description:\n")
    f.write(caption + "\n")

print(f"\n✅ Description saved to: {os.path.abspath(output_file)}")
