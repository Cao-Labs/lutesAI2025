from PIL import Image
from lavis.models import load_model_and_preprocess
import torch
import sys

# Use first command-line argument as image path
if len(sys.argv) < 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", device=device
)

# Load image
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"File not found: {image_path}")
    sys.exit(1)

image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Question/prompt
question = "What is in this image?"

# Generate caption
with torch.no_grad():
    answer = model.generate({"image": image_tensor, "prompt": question})

print("Caption:", answer[0])
