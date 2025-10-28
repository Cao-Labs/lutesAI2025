import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

def main():
    parser = argparse.ArgumentParser(description="Batch caption images using BLIP-2 (LAVIS)")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more image files (space-separated)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on ('cuda' or 'cpu')"
    )
    args = parser.parse_args()

    # Load the model once
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        device=args.device
    )

    for img_path in args.images:
        print(f"\nProcessing: {img_path}")
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: file not found -> {img_path}")
            continue
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Preprocess the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)

        # LAVIS expects the input dict key to be "image"
        with torch.no_grad():
            caption = model.generate({"image": image_tensor})

        print(f"Caption: {caption[0]}")

if __name__ == "__main__":
    main()
