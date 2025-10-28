#!/usr/bin/env python3
import argparse
import os
from PIL import Image
from lavis.models import load_model_and_preprocess

def main():
    parser = argparse.ArgumentParser(description="Run BLIP-2 on an image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    args = parser.parse_args()

    image_path = args.image

    # Debug: print image path
    print(f"[DEBUG] Image path provided: {image_path}")
    print(f"[DEBUG] Does the file exist? {os.path.exists(image_path)}")

    # Load image
    try:
        raw_image = Image.open(image_path).convert("RGB")
        print("[DEBUG] Image loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Image file not found: {image_path}")
        return

    # Load BLIP-2 model
    print("[DEBUG] Loading BLIP-2 model...")
    model, vis_processors, _ = load_model_and_preprocess(
        model_name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True
    )
    print("[DEBUG] Model loaded successfully.")

    # Preprocess image
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    print("[DEBUG] Image preprocessed.")

    # Generate caption
    caption = model.generate({"image": image})
    print("Caption:", caption[0])

if __name__ == "__main__":
    main()
