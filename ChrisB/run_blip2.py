# run_blip2.py

import sys
import os
from pathlib import Path

print(f"Running script from: {__file__}")

def verify_file(path):
    abs_path = os.path.abspath(path)
    print(f"\nFile verification for: {path}")
    print(f"1. Absolute path: {abs_path}")
    print(f"2. Path exists (os.path): {os.path.exists(abs_path)}")
    
    p = Path(path)
    print(f"3. Path object: {p}")
    print(f"4. Resolved path: {p.resolve()}")
    print(f"5. Path exists (pathlib): {p.exists()}")
    
    try:
        if p.exists():
            print(f"6. Is file: {p.is_file()}")
            print(f"7. File size: {p.stat().st_size}")
            with open(p, 'rb') as f:
                print("8. File can be opened")
    except Exception as e:
        print(f"Error during verification: {e}")
    print("="*50)

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

import argparse

# Set up command line arguments first, before any other imports
parser = argparse.ArgumentParser(description='Generate protein function description using BLIP-2')
parser.add_argument('--image', type=str, required=True, help='Path to the protein image file')
args = parser.parse_args()

# Verify the file exists before importing anything else
verify_file(args.image)

from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Load BLIP-2 model
print("Loading BLIP-2 model...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

print("\nOpening image...")
image_path = Path(args.image).resolve()
raw_image = Image.open(str(image_path)).convert("RGB")
print(f"Image opened successfully, size: {raw_image.size}")

# Preprocess and move image to device
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate caption
print("\nGenerating caption...")
caption = model.generate({"image": image})[0]
print("\n🧬 Generated Protein Function Description:\n", caption)
