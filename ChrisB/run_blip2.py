import sys
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# Use image from command-line argument
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "protein_image.png"  # fallback

raw_image = Image.open(image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
