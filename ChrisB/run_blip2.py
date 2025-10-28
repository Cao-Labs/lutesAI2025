# run_blip2.py
print("="*80)
print(f"Running script at: {__file__}")
print("="*80)

import sys
import os
from pathlib import Path

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
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Command line args: {sys.argv}")
print("="*80)

import os
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print("="*80)

import argparse
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

# Verify the image exists before proceeding
print("\nVerifying command line argument...")
print(f"Command line args: {sys.argv}")
print(f"Parsed args: {args}")

if not args.image:
    print("No image path provided")
    sys.exit(1)

verify_file(args.image)

# Try to load the image
try:
    image_path = Path(args.image).resolve()
    print(f"\nAttempting to open image at: {image_path}")
    
    # Try direct file access first
    with open(image_path, 'rb') as f:
        print("File opened successfully with built-in open()")
    
    # Now try with PIL
    raw_image = Image.open(str(image_path)).convert("RGB")
    print(f"Image opened with PIL, size: {raw_image.size}")
except Exception as e:
    print(f"Error details: {str(e)}")
    sys.exit(1)

# Preprocess and move image to device
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate caption (natural language description)
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
