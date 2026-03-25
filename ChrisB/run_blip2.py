# run_blip2.py

import argparse
import os
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

# Force CPU for stability (GPU optional but not required)
device = torch.device("cpu")

print("Loading BLIP-2...")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

"""
BLIP-2 (Flan-T5 XL) is a vision-language model trained on natural images.

IMPORTANT:
- It is NOT trained on scientific heatmaps
- Therefore, prompt design is critical for extracting structure

We keep model unchanged (constraint: no pipeline changes)
"""

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

raw_image = Image.open(args.image).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# --------------------------------------------------
# PROMPTS (STRUCTURE-GROUNDED)
# --------------------------------------------------

"""
Improved prompts:
- Force model to interpret structure (not generic captioning)
- Focus on diagonal patterns, domains, and interactions
"""

PROMPTS = [
    "Describe structural patterns in this protein similarity matrix, focusing on diagonal regions, domain blocks, and interaction clusters.",
    "Identify repeating motifs, symmetry, and irregular regions in this protein matrix and explain their structural meaning.",
    "Explain how diagonal continuity and off-diagonal signals reflect protein domains and interactions in this matrix."
]

# --------------------------------------------------
# FILTER FUNCTION
# --------------------------------------------------

def is_good(text):
    """
    Filters weak or generic captions.

    Criteria:
    - sufficient length
    - lexical diversity
    - presence of structural/biological keywords
    """

    words = text.lower().split()

    if len(words) < 50 or len(set(words)) < 20:
        return False

    keywords = ["domain", "structure", "pattern", "region", "interaction"]
    if not any(k in text.lower() for k in keywords):
        return False

    return True

# --------------------------------------------------
# SCORING FUNCTION (BETTER THAN LENGTH)
# --------------------------------------------------

def score_caption(text):
    """
    Scores captions based on:
    - length
    - presence of structural keywords

    This replaces naive 'longest caption wins'
    """

    score = 0

    # reward length
    score += len(text.split())

    # reward key structural terms
    keywords = ["domain", "structure", "interaction", "motif", "region"]
    score += sum(5 for k in keywords if k in text.lower())

    return score

# --------------------------------------------------
# GENERATION FUNCTION
# --------------------------------------------------

def generate_caption(prompt):
    """
    Generate caption using BLIP-2.

    Tuned for:
    - moderate diversity
    - reduced hallucination
    """

    output = model.generate(
        {"image": image, "prompt": prompt},
        use_nucleus_sampling=True,
        top_p=0.9,
        temperature=0.75,          # reduced from 0.85 → more stable
        repetition_penalty=1.5,
        max_length=150,
        min_length=50
    )

    return output[0].strip()

# --------------------------------------------------
# GENERATE CANDIDATES (SELF-CONSISTENCY)
# --------------------------------------------------

print("Generating candidates...")

candidates = []

for p in PROMPTS:
    for _ in range(2):  # self-consistency sampling
        caption = generate_caption(p)
        if caption:
            candidates.append(caption)

# --------------------------------------------------
# FILTER + SELECT BEST CAPTION
# --------------------------------------------------

good = [c for c in candidates if is_good(c)]

if good:
    final_caption = max(good, key=score_caption)
else:
    final_caption = max(candidates, key=score_caption)

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\nFINAL DESCRIPTION:\n")
print(final_caption)

out_file = args.image.rsplit(".", 1)[0] + "_description.txt"

with open(out_file, "w") as f:
    f.write(final_caption)

print(f"\nSaved → {out_file}")
