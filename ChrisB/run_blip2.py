import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

def main():
    parser = argparse.ArgumentParser(description="Generate captions for one or more images using BLIP-2")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="Paths to image files (space separated)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on: 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Load BLIP-2 model once
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xl", device=args.device
    )

    for img_path in args.images:
        try:
            print(f"Processing: {img_path}")
            raw_image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: File not found at {img_path}")
            continue
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Preprocess and move to device
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)

        # Generate caption
        with torch.no_grad():
            caption = model.generate({"image": image_tensor})
        
        print(f"Caption for {img_path}: {caption[0]}")

if __name__ == "__main__":
    main()
