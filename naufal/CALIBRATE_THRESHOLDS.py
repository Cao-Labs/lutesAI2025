#!/usr/bin/env python3
"""
Find per-label thresholds τ_j that maximize F_beta (default beta=2) on a validation set.
Optionally also compute a protein-adaptive percentile rule.

Outputs:
  - thresholds.npy  (shape [C])
  - calib_report.json (F2 per label, global metrics)
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BigBirdModel
from collections import defaultdict

# ----- Reuse a minimal Dataset consistent with training -----
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
            elif line.startswith("is_a: ") and current_id:
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

class ValProteinDataset(Dataset):
    def __init__(self, embedding_dir, go_mapping_file, go_vocab_path, obo_path, max_len=None, input_dim=None):
        self.embedding_dir = embedding_dir
        self.max_len = max_len
        self.go_graph = extract_go_graph(obo_path)

        with open(go_vocab_path, "r") as f:
            vocab = json.load(f)
        self.go_vocab = vocab if isinstance(vocab, dict) else {go:i for i,go in enumerate(vocab)}
        self.num_labels = len(self.go_vocab)

        all_ids = [fn[:-3] for fn in os.listdir(embedding_dir) if fn.endswith(".pt")]
        self.ids = set(all_ids)

        self.labels = {}
        with open(go_mapping_file, "r") as f:
            for line in f:
                pid, terms = line.rstrip("\n").split("\t")
                if pid in self.ids:
                    term_list = [t.strip() for t in terms.split(";") if t.strip()]
                    full = propagate_terms(term_list, self.go_graph)
                    self.labels[p] = 1  # avoid KeyError
                    self.labels[pid] = [t for t in full if t in self.go_vocab]

        self.ids = sorted(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))
        if self.max_len is not None:
            L = emb.size(0)
            if L > self.max_len:
                emb = emb[:self.max_len]
            elif L < self.max_len:
                pad = torch.zeros(self.max_len - L, emb.size(1), dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
        attn = (emb.sum(dim=1) != 0).long()
        y = torch.zeros(self.num_labels)
        for t in self.labels[pid]:
            y[self.go_vocab[t]] = 1.0
        return pid, emb, attn, y

class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim):
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
        out = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

def fbeta_score(y_true, y_pred, beta=2.0, eps=1e-12):
    # y*: [N,C] numpy {0,1}
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + eps)
    return f, prec, rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dir", required=True)
    ap.add_argument("--go_mapping_file", required=True)
    ap.add_argument("--go_vocab_path", required=True)
    ap.add_argument("--obo_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_dim", type=int, default=1541)
    ap.add_argument("--max_len", type=int, default=1913)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--grid", type=str, default="0.01,0.99,99", help="start,end,num for np.linspace")
    ap.add_argument("--out_thresholds", default="thresholds.npy")
    ap.add_argument("--out_report", default="calib_report.json")
    args = ap.parse_args()

    gstart, gend, gnum = args.grid.split(",")
    grid = np.linspace(float(gstart), float(gend), int(gnum))

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    ds = ValProteinDataset(args.embedding_dir, args.go_mapping_file,
                           args.go_vocab_path, args.obo_path,
                           max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    C = ds.num_labels

    model = BigBirdProteinModel(args.input_dim, C).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # collect probs and labels
    all_probs = []
    all_true = []
    with torch.no_grad():
        for _, x, attn, y in tqdm(dl, desc="[INFO] Running validation forward"):
            x, attn = x.to(device), attn.to(device)
            logits = model(x, attn)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.numpy())
    P = np.concatenate(all_probs, axis=0)  # [N,C]
    Y = np.concatenate(all_true, axis=0)   # [N,C]

    taus = np.zeros(C, dtype=np.float32)
    f2_per_label = np.zeros(C, dtype=np.float32)

    for j in range(C):
        best_f = -1.0
        best_t = 0.5
        pj = P[:, j]
        yj = Y[:, j]
        for t in grid:
            pred = (pj >= t).astype(np.int32)
            f, _, _ = fbeta_score(yj.reshape(-1,1), pred.reshape(-1,1), beta=args.beta)
            if f[0] > best_f:
                best_f = f[0]
                best_t = t
        taus[j] = best_t
        f2_per_label[j] = best_f

    # global F2 with these thresholds
    preds = (P >= taus[None, :]).astype(np.int32)
    f2_global, prec_global, rec_global = fbeta_score(Y, preds, beta=args.beta)
    f2_macro = np.nanmean(f2_per_label)

    np.save(args.out_thresholds, taus)
    report = {
        "beta": args.beta,
        "macro_F2": float(f2_macro),
        "global_precision_mean": float(np.nanmean(prec_global)),
        "global_recall_mean": float(np.nanmean(rec_global))
    }
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[✓] Saved per-label thresholds to {args.out_thresholds}")
    print(f"[✓] Wrote calibration report to {args.out_report}")
    print(f"[INFO] Macro F2={report['macro_F2']:.4f}, "
          f"P={report['global_precision_mean']:.4f}, R={report['global_recall_mean']:.4f}")

if __name__ == "__main__":
    main()
