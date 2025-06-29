import os
import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig
from tqdm import tqdm

# === Paths and Config ===
TEST_DIR = "/data/summer2020/naufal/testing_pca"
GO_MAPPING_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"
MODEL_PATH = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt"
OUTPUT_FILE = "/data/summer2020/naufal/test_pred.txt"

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
INPUT_DIM = 512
MAX_LEN = 512

# === Load GO vocabulary from training mapping file ===
go_terms_set = set()
with open(GO_MAPPING_FILE, "r") as f:
    for line in f:
        try:
            _, terms = line.strip().split("\t")
            go_terms_set.update(terms.split(";"))
        except:
            continue
go_vocab = {idx: term for idx, term in enumerate(sorted(go_terms_set))}
NUM_LABELS = len(go_vocab)

# === Define the model class ===
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

# === Load trained model ===
model = BigBirdProteinModel(INPUT_DIM, NUM_LABELS, MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Prediction loop ===
with open(OUTPUT_FILE, "w") as out_f:
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".pt")])
    for idx, fname in enumerate(tqdm(files, desc="Predicting")):
        pid = fname[:-3]
        path = os.path.join(TEST_DIR, fname)

        try:
            embedding = torch.load(path).to(DEVICE)  # [512, 512]
            if embedding.dim() != 2 or embedding.shape != (512, 512):
                print(f"[!] Skipped {pid}: invalid shape {embedding.shape}")
                continue

            attention_mask = (embedding.sum(dim=1) != 0).long()
            embedding = embedding.unsqueeze(0)        # [1, 512, 512]
            attention_mask = attention_mask.unsqueeze(0)  # [1, 512]

            with torch.no_grad():
                logits = model(embedding, attention_mask)
                probs = torch.sigmoid(logits)[0]
                predicted_indices = (probs > 0.5).nonzero(as_tuple=True)[0]
                predicted_terms = [go_vocab[i.item()] for i in predicted_indices]

            out_f.write(f"{pid}\t{';'.join(predicted_terms)}\n")

            if idx == 0 or (idx + 1) % 10000 == 0:
                print(f"[✓] Wrote predictions for {idx + 1:,} proteins")

        except Exception as e:
            print(f"[!] Failed on {pid}: {e}")

print(f"[✓] All predictions saved to {OUTPUT_FILE}")
