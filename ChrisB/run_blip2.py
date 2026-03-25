# run_blip2.py

import argparse
import os
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

device = torch.device("cpu")

print("Loading BLIP-2...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

raw_image = Image.open(args.image).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# 🔥 MULTI-PROMPT SET
PROMPTS = [
    "Describe structural domains and interactions in this protein matrix.",
    "Identify repeating motifs and unusual regions in this protein.",
    "Explain diagonal and off-diagonal patterns in this matrix."
]


def is_good(text):
    words = text.split()
    return len(set(words)) > 15 and len(words) > 40


def generate_caption(prompt):
    output = model.generate(
        {"image": image, "prompt": prompt},
        use_nucleus_sampling=True,
        top_p=0.9,
        temperature=0.85,
        repetition_penalty=1.5,
        max_length=150,
        min_length=50
    )
    return output[0].strip()


print("Generating candidates...")

candidates = []
for p in PROMPTS:
    for _ in range(2):  # self-consistency
        caption = generate_caption(p)
        if caption:
            candidates.append(caption)

# 🔥 FILTER BAD OUTPUTS
good = [c for c in candidates if is_good(c)]

if good:
    final_caption = max(good, key=len)  # longest = most informative
else:
    final_caption = max(candidates, key=len)

print("\nFINAL DESCRIPTION:\n")
print(final_caption)

out_file = args.image.rsplit(".", 1)[0] + "_description.txt"
with open(out_file, "w") as f:
    f.write(final_caption)

print(f"\nSaved → {out_file}")
