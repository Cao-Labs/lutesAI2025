# run_blip2.py

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
image_path = "protein_image.png"  # Or path to your actual output image
raw_image = Image.open(image_path).convert("RGB")

# Preprocess and move image to device
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate caption (natural language description)
caption = model.generate({"image": image})[0]
print("🧬 Generated Protein Function Description:\n", caption)
