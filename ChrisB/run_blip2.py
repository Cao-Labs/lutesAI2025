import argparse
from PIL import Image
from lavis.models import load_model_and_preprocess

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate captions for an image using BLIP-2")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on: 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Load and verify image
    try:
        raw_image = Image.open(args.image).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File not found at {args.image}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load BLIP-2 model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xl", device=args.device
    )

    # Preprocess image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(args.device)

    # Generate caption
    with torch.no_grad():
        caption = model.generate({"image": image})
    
    print("Generated Caption:")
    print(caption[0])

if __name__ == "__main__":
    import torch
    main()
