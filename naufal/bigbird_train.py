
# Proteinext 2.0 - BigBird training 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BigBirdModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import json
import csv
from tqdm import tqdm

# === Config ===
DATA_DIR = "/data/summer2020/naufal"
EMBEDDING_DIR = os.path.join(DATA_DIR, "esm3_embeddings")
FEATURE_FILE = os.path.join(DATA_DIR, "protein_features.txt")
GO_LABEL_FILE = os.path.join(DATA_DIR, "matched_ids_with_go.txt")
OUTPUT_DIR = os.path.join(DATA_DIR, "trained_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Load GO terms ===
print("Loading GO term labels...")
protein_to_go = {}
all_go_terms = set()
with open(GO_LABEL_FILE) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, go_str = parts
        terms = go_str.split(";")
        protein_to_go[pid] = terms
        all_go_terms.update(terms)
all_go_terms = sorted(list(all_go_terms))
print(f"Total unique GO terms: {len(all_go_terms)}")

# === Step 2: One-hot encode GO labels ===
mlb = MultiLabelBinarizer(classes=all_go_terms)
protein_ids = []
label_vectors = []
for pid in protein_to_go:
    label_vectors.append(protein_to_go[pid])
    protein_ids.append(pid)
binary_labels = mlb.fit_transform(label_vectors)
labels_tensor = torch.tensor(binary_labels, dtype=torch.float32)

# === Step 3: Load embeddings and concatenate features ===
print("Loading embeddings and secondary structure...")
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

X = []
y = []
for i, pid in enumerate(protein_ids):
    emb_path = os.path.join(EMBEDDING_DIR, f"{pid}.pt")
    if not os.path.exists(emb_path) or pid not in feature_map:
        continue
    emb = torch.load(emb_path)
    feat = torch.tensor(feature_map[pid])
    if emb.dim() == 3:
        emb = emb.squeeze()
    if feat.dim() == 3:
        feat = feat.squeeze()
    if emb.shape[0] != feat.shape[0] or emb.dim() != 2 or feat.dim() != 2:
        continue
    combined = torch.cat([emb, feat], dim=1)
    pooled = combined.mean(dim=0)  # mean-pooling for fixed-size input
    X.append(pooled)
    y.append(labels_tensor[i])
    if (i+1) % 5000 == 0 or (i+1) % 100000 == 0:
        print(f"Processed {i+1} proteins")

X = torch.stack(X)
y = torch.stack(y)

# === Step 4: Create train/test split ===
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

# === Step 5: Define model ===
class CustomBigBirdModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding_proj = nn.Linear(input_dim, hidden_dim)
        self.bigbird = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings):
        x = self.embedding_proj(embeddings)
        x = x.unsqueeze(1)
        output = self.bigbird(inputs_embeds=x)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

# === Step 6: Train function ===
def train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss)
        print(f"Epoch {epoch+1} loss: {running_loss:.4f}")
        model_path = os.path.join(OUTPUT_DIR, f"bigbird_model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model checkpoint saved to: {model_path}")

# === Step 7: Train ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = y.shape[1]
model = CustomBigBirdModel(input_dim=1282, hidden_dim=768, num_classes=num_classes)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=5, device=device)
