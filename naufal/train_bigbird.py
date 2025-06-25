import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BigBirdModel, BigBirdConfig
from tqdm import tqdm
from collections import defaultdict

# === Custom Dataset ===
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
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))

        target = torch.zeros(self.num_labels)
        for term in self.go_labels.get(pid, []):
            if term in self.go_vocab:
                target[self.go_vocab[term]] = 1.0

        attention_mask = (embedding.abs().sum(dim=-1) != 0).float()

        return embedding, attention_mask, target, pid

# === Model ===
class BigBirdClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        config = BigBirdConfig(
            hidden_size=input_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            attention_type="block_sparse",
            block_size=64,
            seq_length=1913,
            vocab_size=1  # Not used
        )
        self.encoder = BigBirdModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x, attention_mask):
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Use first token as CLS
        logits = self.classifier(cls_token)
        return logits

# === Training Loop ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProteinFunctionDataset(
        embedding_dir="/data/archives/naufal/final_embeddings",
        go_mapping_file="/data/summer2020/naufal/matched_ids_with_go.txt"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = BigBirdClassifier(input_dim=1541, num_labels=dataset.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    print("[INFO] Starting training...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0

        for i, (embeddings, attn_mask, targets, ids) in enumerate(dataloader):
            embeddings, attn_mask, targets = embeddings.to(device), attn_mask.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings, attn_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i == 0 or (i + 1) % 10000 == 0:
                print(f"[✓] Epoch {epoch+1} | Batch {i+1:,} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    print("[INFO] Training complete.")

if __name__ == "__main__":
    train()
