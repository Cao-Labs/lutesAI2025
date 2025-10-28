# run_blip2.py

import argparse
import sys
from pathlib import Path

# Set up command line arguments first, before any other imports
parser = argparse.ArgumentParser(description='Generate protein function description using BLIP-2')
parser.add_argument('--image', type=str, required=True, help='Path to the protein image file')
args = parser.parse_args()

# Validate the image path immediately
image_path = Path(args.image).resolve()
if not image_path.exists():
    print(f"Error: Image file not found: {image_path}")
    sys.exit(1)

# Only import the rest if the image exists
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 model (pretrained FLAN-T5 variant)
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# Load protein image generated from ESM-3
print(f"Debug - Command line arguments: {sys.argv}")
print(f"Debug - Parsed args: {args}")
print(f"Debug - Current working directory: {Path.cwd()}")
print(f"Debug - Received image path argument: {args.image}")

try:
    image_path = Path(args.image).resolve()
    print(f"Debug - Resolved image path: {image_path}")
    print(f"Debug - Path exists check: {image_path.exists()}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Debug - Attempting to open image at: {image_path}")
    raw_image = Image.open(str(image_path)).convert("RGB")
except Exception as e:
    print(f"Error details: {str(e)}")
    sys.exit(1)

# Preprocess and move image to device
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate caption (natural language description)
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
