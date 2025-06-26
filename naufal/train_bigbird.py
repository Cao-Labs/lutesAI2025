import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdForSequenceClassification, BigBirdConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from sklearn.metrics import f1_score
from tqdm import tqdm

# === Dataset Class ===
class ProteinFunctionDataset(Dataset):
    def __init__(self, embedding_dir, go_mapping_file):
        self.embedding_dir = embedding_dir
        self.go_mapping_file = go_mapping_file

        print("[INFO] Scanning embedding files...")
        self.ids = set(fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt"))
        print(f"[INFO] Found {len(self.ids):,} embedding files.")

        self.go_labels = defaultdict(list)
        go_terms_set = set()

        print("[INFO] Parsing GO annotations...")
        with open(go_mapping_file, "r") as f:
            for line in f:
                pid, terms = line.strip().split("\t")
                if pid in self.ids:
                    term_list = terms.split(";")
                    self.go_labels[pid] = term_list
                    go_terms_set.update(term_list)

        self.go_vocab = {go_term: idx for idx, go_term in enumerate(sorted(go_terms_set))}
        self.num_labels = len(self.go_vocab)
        print(f"[INFO] GO vocabulary size: {self.num_labels:,}")

        self.ids = list(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [L, 512]

        attention_mask = (embedding.sum(dim=1) != 0).long()  # [L]
        target = torch.zeros(self.num_labels)
        for term in self.go_labels.get(pid, []):
            if term in self.go_vocab:
                target[self.go_vocab[term]] = 1.0

        return embedding, attention_mask, target

# === Model Wrapper ===
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, max_len):
        super().__init__()
        self.project = nn.Linear(input_dim, 1536)  # Make divisible by 8 heads
        config = BigBirdConfig(
            vocab_size=50265,
            hidden_size=1536,
            num_attention_heads=8,
            num_hidden_layers=12,
            attention_type="block_sparse",
            block_size=64,
            max_position_embeddings=max_len,
            use_bias=True,
            is_decoder=False,
            num_labels=target_dim,
        )
        self.bigbird = BigBirdForSequenceClassification(config)
        self.bigbird.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)  # [B, L, 1536]
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        return outputs.logits  # [B, num_labels]

# === Training Function ===
def train():
    # === Device Selection ===
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    # === Training Settings ===
    batch_size = 32
    epochs = 5
    learning_rate = 1e-5
    patience = 2
    min_lr = 1e-7

    dataset = ProteinFunctionDataset(
        "/data/summer2020/naufal/final_embeddings_pca",
        "/data/summer2020/naufal/matched_ids_with_go.txt"
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = BigBirdProteinModel(input_dim=512, target_dim=dataset.num_labels, max_len=512).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, min_lr=min_lr)
    criterion = nn.BCEWithLogitsLoss()

    print("[✓] Starting training...")
    model.train()

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        total_loss = 0.0

        for i, (x, attn_mask, y) in enumerate(tqdm(dataloader)):
            x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x, attn_mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i == 0 or (i + 1) % 10000 == 0:
                print(f"[✓] Trained {i + 1:,} proteins")

        avg_loss = total_loss / len(dataloader)
        print(f"[INFO] Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        torch.save(model.state_dict(), f"bigbird_finetuned_epoch_{epoch+1}.pt")
        print(f"[✓] Model saved: bigbird_finetuned_epoch_{epoch+1}.pt")

    print("[✓] Training complete.")

if __name__ == "__main__":
    train()


