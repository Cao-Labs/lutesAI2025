import os
import json
import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig
from tqdm import tqdm

# === Paths ===
EMBEDDING_DIR = "/data/summer2020/naufal/testing_pca"
GO_VOCAB_PATH = "/data/shared/github/lutesAI2025/naufal/go_vocab.json"
MODEL_PATH = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt"
OUTPUT_PATH = "/data/summer2020/naufal/test_pred.txt"

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

# === Load GO vocab ===
with open(GO_VOCAB_PATH, "r") as f:
    go_vocab = json.load(f)
idx_to_go = {idx: go for go, idx in go_vocab.items()}
num_labels = len(idx_to_go)

# === Load model ===
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
model = BigBirdProteinModel(input_dim=512, target_dim=num_labels, max_len=512).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Run predictions and save ===
sigmoid = nn.Sigmoid()

with open(OUTPUT_PATH, "w") as out_f:
    for fname in tqdm(sorted(os.listdir(EMBEDDING_DIR))):
        if not fname.endswith(".pt"):
            continue

        pid = fname[:-3]
        embedding = torch.load(os.path.join(EMBEDDING_DIR, fname))  # [512, 512]
        attn_mask = (embedding.sum(dim=1) != 0).long()  # [512]

        x = embedding.unsqueeze(0).to(device)  # [1, 512, 512]
        attn_mask = attn_mask.unsqueeze(0).to(device)  # [1, 512]

        with torch.no_grad():
            logits = model(x, attn_mask)  # [1, num_labels]
            probs = sigmoid(logits).squeeze(0)  # [num_labels]

            # === Predict with threshold ===
            threshold = 0.5
            above_thresh = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()

            if above_thresh:
                predicted_terms = [idx_to_go[idx] for idx in above_thresh]
            else:
                # Fallback: top 3 if none above threshold
                topk = 3
                top_indices = torch.topk(probs, k=topk).indices.tolist()
                predicted_terms = [idx_to_go[idx] for idx in top_indices]

        out_f.write(f"{pid}\t{';'.join(predicted_terms)}\n")
