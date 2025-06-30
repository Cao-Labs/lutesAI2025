import os
import json
import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig
from tqdm import tqdm

# === Config ===
EMBEDDING_DIR = "/data/summer2020/naufal/testing_pca"
VOCAB_FILE = "/data/shared/github/lutesAI2025/naufal/go_vocab.json"
MODEL_FILE = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt"
OUTPUT_FILE = "/data/summer2020/naufal/test_pred.txt"
MAX_LEN = 512
INPUT_DIM = 512
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

# === Load GO vocab ===
with open(VOCAB_FILE, "r") as f:
    go_vocab = json.load(f)

idx_to_go = {v: k for k, v in go_vocab.items()}
num_labels = len(go_vocab)

# === Model Definition ===
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, max_len):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)
        config = BigBirdConfig(
            vocab_size=50265,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            attention_type="block_sparse",
            block_size=64,
            max_position_embeddings=max_len,
            use_bias=True,
            is_decoder=False,
        )
        self.bigbird = BigBirdModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# === Load model ===
model = BigBirdProteinModel(INPUT_DIM, num_labels, MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# === Predict and write output ===
with open(OUTPUT_FILE, "w") as out_f:
    files = sorted(f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".pt"))
    print(f"[INFO] Found {len(files)} test embedding files")

    for i, fname in enumerate(tqdm(files)):
        pid = fname[:-3]
        embed = torch.load(os.path.join(EMBEDDING_DIR, fname)).to(DEVICE)
        attn_mask = (embed.sum(dim=1) != 0).long().to(DEVICE)

        with torch.no_grad():
            logits = model(embed.unsqueeze(0), attn_mask.unsqueeze(0))
            preds = torch.sigmoid(logits).squeeze().cpu()
            go_indices = (preds >= 0.5).nonzero(as_tuple=True)[0].tolist()
            go_terms = [idx_to_go[idx] for idx in go_indices if idx in idx_to_go]

        out_f.write(f"{pid}\t{';'.join(go_terms)}\n")

        if i == 0 or (i + 1) % 1000 == 0:
            print(f"[✓] Predicted {i + 1} proteins")

print(f"[✓] Predictions saved to {OUTPUT_FILE}")

