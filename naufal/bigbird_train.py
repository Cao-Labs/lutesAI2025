# Proteinext 2.0 - Debugged, Filtered, and Verbose BigBird Training Script

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BigBirdModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict

# === Config ===
DATA_DIR = "/data/summer2020/naufal"
EMBEDDING_DIR = os.path.join(DATA_DIR, "esm3_embeddings")
FEATURE_FILE = os.path.join(DATA_DIR, "protein_features.txt")
GO_LABEL_FILE = os.path.join(DATA_DIR, "matched_ids_with_go.txt")
OBO_FILE = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
OUTPUT_DIR = os.path.join(DATA_DIR, "trained_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === GO Utilities ===
def parse_go_obo(obo_path):
    go_info = {}
    current = {}
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current = {}
            elif line == "":
                if 'id' in current:
                    go_info[current['id']] = {
                        'name': current.get('name', ''),
                        'namespace': current.get('namespace', ''),
                        'definition': current.get('def', ''),
                        'parents': current.get('is_a', [])
                    }
            elif line.startswith('id:'):
                current['id'] = line.split('id: ')[1]
            elif line.startswith('name:'):
                current['name'] = line.split('name: ')[1]
            elif line.startswith('namespace:'):
                current['namespace'] = line.split('namespace: ')[1]
            elif line.startswith('def:'):
                current['def'] = line.split('def: ')[1]
            elif line.startswith('is_a:'):
                if 'is_a' not in current:
                    current['is_a'] = []
                parent_id = line.split('is_a: ')[1].split(' !')[0]
                current['is_a'].append(parent_id)
    return go_info

# === Load GO Labels ===
print("Loading GO term labels...")
protein_to_go = {}
all_go_terms = set()
with open(GO_LABEL_FILE) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, go_str = parts
        if go_str == "NA":
            continue
        terms = go_str.split(";")
        protein_to_go[pid] = terms
        all_go_terms.update(terms)
all_go_terms = sorted(list(all_go_terms))
mlb = MultiLabelBinarizer(classes=all_go_terms)
binary_labels = mlb.fit_transform([protein_to_go[pid] for pid in protein_to_go])
label_map = {pid: binary_labels[i] for i, pid in enumerate(protein_to_go)}

# === Load Secondary Structure ===
print("Loading secondary structure features...")
feature_map = {}
current_id = None
current_features = []
with open(FEATURE_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and current_features:
                feature_map[current_id] = current_features
            current_id = line[1:].strip()
            current_features = []
        elif line:
            parts = line.split()
            current_features.append([float(p) for p in parts[1:]])
if current_id and current_features:
    feature_map[current_id] = current_features

# === Dataset ===
class ProteinDataset(Dataset):
    def __init__(self, ids, label_map, feature_map, embedding_dir, input_dim):
        self.ids = ids
        self.labels = label_map
        self.feature_map = feature_map
        self.embedding_dir = embedding_dir
        self.input_dim = input_dim

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb_path = os.path.join(self.embedding_dir, f"{pid}.pt")
        if not os.path.exists(emb_path) or pid not in self.feature_map:
            return None

        emb = torch.load(emb_path)
        feat = torch.tensor(self.feature_map[pid])
        if emb.dim() == 3:
            emb = emb.squeeze()
        if feat.dim() == 3:
            feat = feat.squeeze()
        if emb.shape[0] != feat.shape[0] or emb.dim() != 2 or feat.dim() != 2:
            return None

        x = torch.cat([emb, feat], dim=1)
        y = torch.tensor(self.labels[pid], dtype=torch.float32)
        return x, y

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x_list, y_list = zip(*batch)
    max_len = max([x.shape[0] for x in x_list])
    x_padded = []
    mask_padded = []
    for x in x_list:
        pad_len = max_len - x.shape[0]
        x_pad = F.pad(x, (0, 0, 0, pad_len))
        mask = torch.cat([torch.ones(x.shape[0]), torch.zeros(pad_len)])
        x_padded.append(x_pad)
        mask_padded.append(mask)
    return torch.stack(x_padded), torch.stack(mask_padded), torch.stack(y_list)

# === Model ===
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)

class ProteinBigBirdAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.bigbird = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
        self.attn_pool = AttentionPooling(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        x = self.proj(x)
        x = x.to(torch.float32)
        x = x.to(next(self.parameters()).device)
        mask = mask.to(torch.long)
        out = self.bigbird(inputs_embeds=x, attention_mask=mask)
        pooled = self.attn_pool(out.last_hidden_state, mask)
        return self.classifier(pooled)

# === Prepare Data
print("Preparing dataset...")
all_ids = list(protein_to_go.keys())
train_ids, test_ids = train_test_split(all_ids, test_size=0.3, random_state=42)
train_dataset = ProteinDataset(train_ids, label_map, feature_map, EMBEDDING_DIR, 1282)
test_dataset = ProteinDataset(test_ids, label_map, feature_map, EMBEDDING_DIR, 1282)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# === Train Function
def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if batch is None:
                continue
            x, mask, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"Step {i+1}: Loss = {loss.item():.6f}")
            print(f"logits shape: {logits.shape}, y shape: {y.shape}")

        scheduler.step(running_loss)
        print(f"Epoch {epoch+1} loss: {running_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"debug_bigbird_epoch{epoch+1}.pt"))

# === Evaluation
def evaluate_model(model, dataloader, device, mlb, output_csv):
    print("Evaluating model and exporting predictions...")
    model.eval()
    model.to(device)
    predictions = []
    true_labels = []
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            x, mask, y = [b.to(device) for b in batch]
            logits = model(x, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            predictions.extend(preds)
            true_labels.extend(y.cpu().numpy())

            for p in probs:
                if total == 0 or total % 50000 == 0:
                    print(f"Prediction {total + 1} written")
                total += 1

    precision = precision_score(true_labels, predictions, average='samples', zero_division=0)
    recall = recall_score(true_labels, predictions, average='samples', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='samples', zero_division=0)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    ids = list(protein_to_go.keys())[-len(predictions):]
    csv_path = os.path.join(OUTPUT_DIR, output_csv)
    with open(csv_path, "w") as f:
        f.write("ProteinID,PredictedGO
")
        for pid, pred in zip(ids, predictions):
            terms = [mlb.classes_[j] for j in range(len(pred)) if pred[j] == 1]
            f.write(f"{pid},{';'.join(terms)}\n")
    print(f"CSV export complete: {csv_path}")

# === Start training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ProteinBigBirdAttention(input_dim=1282, hidden_dim=768, num_classes=len(all_go_terms))
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=5, device=device)
evaluate_model(model, test_loader, device, mlb, output_csv="predictions.csv")
