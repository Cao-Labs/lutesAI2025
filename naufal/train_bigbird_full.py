import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from tqdm import tqdm

def extract_go_graph(obo_path):
    go_graph = defaultdict(set)
    current_id = None
    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a:") and current_id:
                parent = line.split("is_a: ")[1].split()[0]
                go_graph[current_id].add(parent)
    return go_graph

def propagate_terms(go_terms, go_graph):
    visited = set()
    stack = list(go_terms)
    while stack:
        term = stack.pop()
        if term not in visited:
            visited.add(term)
            stack.extend(go_graph.get(term, []))
    return visited

class ProteinFunctionDataset(Dataset):
    def __init__(self, embedding_dir, go_mapping_file, go_graph):
        self.embedding_dir = embedding_dir
        self.go_graph = go_graph

        print("[INFO] Scanning embedding files...")
        self.ids = set(fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt"))
        print(f"[INFO] Found {len(self.ids):,} embedding files.")

        self.go_labels = defaultdict(list)
        go_terms_set = set()

        print("[INFO] Parsing GO annotations and propagating...")
        with open(go_mapping_file, "r") as f:
            for line in f:
                pid, terms = line.strip().split("\t")
                if pid in self.ids:
                    term_list = [t.strip() for t in terms.split(";") if t.strip()]
                    full_terms = propagate_terms(term_list, go_graph)
                    self.go_labels[pid] = list(full_terms)
                    go_terms_set.update(full_terms)

        self.go_vocab = {go_term: idx for idx, go_term in enumerate(sorted(go_terms_set))}
        self.num_labels = len(self.go_vocab)
        print(f"[INFO] GO vocabulary size after propagation: {self.num_labels:,}")
        self.ids = list(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [1913, 1541]
        attention_mask = (embedding.sum(dim=1) != 0).long()
        target = torch.zeros(self.num_labels)
        for term in self.go_labels.get(pid, []):
            if term in self.go_vocab:
                target[self.go_vocab[term]] = 1.0
        return embedding, attention_mask, target

class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, max_len):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)
        self.bigbird = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base",
            attention_type="block_sparse"
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

def train():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    batch_size = 4
    epochs = 5
    learning_rate = 3e-5  # ⬆️ Slightly increased
    min_lr = 1e-7

    obo_path = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
    embedding_dir = "/data/archives/naufal/final_embeddings"
    go_mapping_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
    vocab_output_path = "/data/shared/github/lutesAI2025/naufal/go_vocab_full.json"
    model_output_path = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned_full.pt"

    print("[INFO] Parsing GO DAG...")
    go_graph = extract_go_graph(obo_path)

    print("[INFO] Loading dataset...")
    dataset = ProteinFunctionDataset(embedding_dir, go_mapping_file, go_graph)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = BigBirdProteinModel(input_dim=1541, target_dim=dataset.num_labels, max_len=1913).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=min_lr)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        epoch_loss = 0.0
        for i, (x, attn_mask, y) in enumerate(tqdm(dataloader)):
            x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = model(x, attn_mask)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔒 Clip gradients
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if i == 0 or (i + 1) % 500 == 0:
                print(f"[✓] Trained {i + 1:,} proteins")

        avg_loss = epoch_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

        # 👁 Log one sample prediction
        model.eval()
        with torch.no_grad():
            sample = dataset[0]
            sx, smask, _ = sample[0].unsqueeze(0).to(device), sample[1].unsqueeze(0).to(device), sample[2]
            sout = torch.sigmoid(model(sx, smask))
            print("[DEBUG] Sample prediction probs (first 10):", sout.squeeze()[:10].cpu().numpy())
        model.train()

    torch.save(model.state_dict(), model_output_path)
    print(f"[✓] Model saved to {model_output_path}")

    with open(vocab_output_path, "w") as f:
        json.dump(dataset.go_vocab, f)
    print(f"[✓] Saved GO vocabulary to {vocab_output_path}")

if __name__ == "__main__":
    train()



