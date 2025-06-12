# Proteinext 2.0 Training Script with all fixes


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel, BigBirdConfig
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import csv

# Configuration
DATA_DIR = "/data/summer2020/naufal"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
INPUT_DIM = 1282
HIDDEN_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OBO_FILE = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"

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

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)

class ProteinBigBirdWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super().__init__()
        config = BigBirdConfig(
            vocab_size=50000,
            hidden_size=hidden_dim,
            num_attention_heads=8,
            num_hidden_layers=6,
            attention_type="block_sparse",
            max_position_embeddings=MAX_SEQ_LEN,
            num_labels=num_labels,
        )
        self.encoder = BigBirdModel(config)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, attention_mask):
        outputs = self.encoder(inputs_embeds=x, attention_mask=attention_mask)
        pooled = self.attn_pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)

class ProteinFunctionDataset(Dataset):
    def __init__(self, matched_ids_file, embeddings_dir, features_file, label_vocab):
        self.matched = []
        self.go_vocab = label_vocab
        self.embeddings_dir = embeddings_dir
        self.feature_map = self.load_secondary_structure(features_file)

        with open(matched_ids_file) as f:
            for line in f:
                pid, go_str = line.strip().split("\t")
                go_terms = go_str.split(";") if go_str != "NA" else []
                self.matched.append((pid, go_terms))

    def load_secondary_structure(self, path):
        feature_map = {}
        current_id = None
        current_features = []

        with open(path) as f:
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
        return feature_map

    def __len__(self):
        return len(self.matched)

    def __getitem__(self, idx):
        pid, go_terms = self.matched[idx]
        emb_path = os.path.join(self.embeddings_dir, f"{pid}.pt")
        if not os.path.exists(emb_path) or pid not in self.feature_map:
            return None

        embeddings = torch.load(emb_path)
        features = torch.tensor(self.feature_map[pid])
        if embeddings.shape[0] != features.shape[0]:
            return None

        x = torch.cat([embeddings, features], dim=1)
        length = x.shape[0]
        if length > MAX_SEQ_LEN:
            x = x[:MAX_SEQ_LEN]
            mask = torch.ones(MAX_SEQ_LEN)
        else:
            pad_len = MAX_SEQ_LEN - length
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])

        y = torch.zeros(len(self.go_vocab))
        for term in go_terms:
            if term in self.go_vocab:
                y[self.go_vocab[term]] = 1.0

        return x, mask, y

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x, mask, y = zip(*batch)
    return torch.stack(x), torch.stack(mask), torch.stack(y)

def build_go_graph(go_dict):
    graph = defaultdict(list)
    for term, data in go_dict.items():
        for parent in data.get("parents", []):
            graph[term].append(parent)
    return graph

def semantic_distance(y_true, y_pred, graph):
    s_values = []
    for true_vec, pred_vec in zip(y_true, y_pred):
        true_terms = {i for i, v in enumerate(true_vec) if v == 1}
        pred_terms = {i for i, v in enumerate(pred_vec) if v == 1}
        union = true_terms | pred_terms
        if not union:
            continue
        disjoint = true_terms ^ pred_terms
        s_val = len(disjoint) / len(union)
        s_values.append(s_val)
    return np.mean(s_values) if s_values else 1.0

def evaluate(model, dataloader, go_graph, threshold=0.5, go_vocab=None):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x, mask, y = [b.to(DEVICE) for b in batch]
            logits = model(x, mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    binarized_preds = (all_preds > threshold).astype(int)

    precision = precision_score(all_true, binarized_preds, average='samples', zero_division=0)
    recall = recall_score(all_true, binarized_preds, average='samples', zero_division=0)
    f1 = f1_score(all_true, binarized_preds, average='samples', zero_division=0)

    smin = semantic_distance(all_true, binarized_preds, go_graph)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Smin: {smin:.4f}")
    return precision, recall, f1, smin

def save_predictions_to_csv(model, dataloader, go_vocab, output_path, threshold=0.5):
    model.eval()
    pred_list = []
    inv_vocab = {v: k for k, v in go_vocab.items()}

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x, mask, _ = [b.to(DEVICE) for b in batch]
            logits = model(x, mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            for p in probs:
                if len(pred_list) == 5000 or len(pred_list) % 100000 == 0:
                    print(f"Predicted for {len(pred_list)} proteins...")
                pred_terms = [inv_vocab[i] for i in range(len(p)) if p[i] > threshold]
                pred_list.append(";".join(pred_terms))

    matched_file = os.path.join(DATA_DIR, "matched_ids_with_go.txt")
    ids = []
    with open(matched_file) as f:
        for line in f:
            pid = line.strip().split("\t")[0]
            ids.append(pid)

    with open(os.path.join(DATA_DIR, "proteinext_predictions.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Protein_ID", "Predicted_GO_Terms"])
        for pid, preds in zip(ids[:len(pred_list)], pred_list):
            writer.writerow([pid, preds])

    print(f"Predictions saved to {os.path.join(DATA_DIR, 'proteinext_predictions.csv')}")

def train():
    matched_file = os.path.join(DATA_DIR, "matched_ids_with_go.txt")
    go_vocab = {}
    with open(matched_file) as f:
        for line in f:
            _, go_str = line.strip().split("\t")
            if go_str != "NA":
                for term in go_str.split(";"):
                    if term not in go_vocab:
                        go_vocab[term] = len(go_vocab)

    go_info = parse_go_obo(OBO_FILE)
    go_graph = build_go_graph(go_info)

raw_dataset = ProteinFunctionDataset(
    matched_file,
    os.path.join(DATA_DIR, "esm3_embeddings"),
    os.path.join(DATA_DIR, "protein_features.txt"),
    go_vocab
)

valid_ids = [x[0] for x in raw_dataset if x is not None]
print(f"Total valid proteins in dataset: {len(valid_ids)}")

train_ids, test_ids = train_test_split(valid_ids, test_size=0.3, random_state=42)

    )
    full_dataset = ProteinFunctionDataset(matched_file, os.path.join(DATA_DIR, "esm3_embeddings"), os.path.join(DATA_DIR, "protein_features.txt"), go_vocab)
    train_dataset = [full_dataset[i] for i in range(len(full_dataset)) if full_dataset[i] is not None and full_dataset.matched[i][0] in train_ids]
    test_dataset = [full_dataset[i] for i in range(len(full_dataset)) if full_dataset[i] is not None and full_dataset.matched[i][0] in test_ids]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ProteinBigBirdWithAttention(INPUT_DIM, HIDDEN_DIM, len(go_vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        model.train()
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
            x, mask, y = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
        print(f"Evaluating after epoch {epoch}:")
        evaluate(model, test_loader, go_graph, go_vocab=go_vocab)

        torch.save(model.state_dict(), os.path.join(DATA_DIR, f"proteinext_bigbird_epoch{epoch}.pt"))

    save_predictions_to_csv(model, test_loader, go_vocab, DATA_DIR)

if __name__ == "__main__":
    train()
