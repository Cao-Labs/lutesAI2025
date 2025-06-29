import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel, BigBirdConfig
from tqdm import tqdm

# === Constants ===
TEST_EMBEDDINGS_DIR = "/data/summer2020/naufal/testing_pca"
GO_VOCAB_FILE = "go_vocab.json"  # must be saved from training phase
MODEL_PATH = "bigbird_finetuned.pt"
OUTPUT_FILE = "test_pred.txt"

# === Dataset for Prediction ===
class TestProteinDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        self.ids = sorted(fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [512, 512]
        attention_mask = (embedding.sum(dim=1) != 0).long()  # [512]
        return pid, embedding, attention_mask

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
        x = self.project(x)  # [B, L, 768]
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(cls_output)  # [B, num_labels]
        return logits

# === Prediction Function ===
def predict():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")

    # === Load GO vocab used during training ===
    with open(GO_VOCAB_FILE, "r") as f:
        go_vocab = json.load(f)
    idx_to_go = {v: k for k, v in go_vocab.items()}
    num_labels = len(go_vocab)
    print(f"[✓] Loaded GO vocab with {num_labels} terms")

    # === Dataset and Model ===
    dataset = TestProteinDataset(TEST_EMBEDDINGS_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = BigBirdProteinModel(input_dim=512, target_dim=num_labels, max_len=512).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[✓] Loaded trained model")

    with open(OUTPUT_FILE, "w") as out_f:
        with torch.no_grad():
            for i, (pid, embedding, attn_mask) in enumerate(tqdm(dataloader)):
                embedding = embedding.to(device)
                attn_mask = attn_mask.to(device)
                logits = model(embedding, attn_mask)
                preds = torch.sigmoid(logits).squeeze(0)  # [num_labels]

                # Thresholding at 0.5
                predicted_terms = [idx_to_go[idx] for idx, score in enumerate(preds) if score.item() >= 0.5]
                out_f.write(f"{pid[0]}\t{';'.join(predicted_terms)}\n")

                if i == 0 or (i + 1) % 1000 == 0:
                    print(f"[✓] Predicted {i + 1:,} proteins")

    print(f"[✓] All predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    predict()
