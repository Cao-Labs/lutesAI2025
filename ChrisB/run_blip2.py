
---

### 📜 `run_blip2.py` (Image → Caption script)

```python
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 model and processors
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)

# Load protein image
raw_image = Image.open("protein_image.png").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# Generate description
caption = model.generate({"image": image})
print("Generated Description:", caption)
