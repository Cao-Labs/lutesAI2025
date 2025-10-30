#!/usr/bin/env python3
import os
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import math
import numpy as np

# ----------------------------
# GO utilities
# ----------------------------
def extract_go_graph_and_ns(obo_path):
    """
    Parse GO OBO to build:
      - go_graph: child -> set(parents)
      - go_namespace: GO -> 'biological_process'|'molecular_function'|'cellular_component'
    """
    go_graph = defaultdict(set)
    go_namespace = {}
    current_id = None
    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif current_id and line.startswith("is_a: "):
                parent = line.split("is_a: ")[1].split()[0]
                go_graph[current_id].add(parent)
            elif current_id and line.startswith("namespace: "):
                ns = line.split("namespace: ")[1]
                go_namespace[current_id] = ns
    return go_graph, go_namespace

def propagate_terms(go_terms, go_graph):
    """Add ancestors (closure)."""
    visited = set()
    stack = list(go_terms)
    while stack:
        term = stack.pop()
        if term not in visited:
            visited.add(term)
            stack.extend(go_graph.get(term, []))
    return visited

# ----------------------------
# Dataset
# ----------------------------
class ProteinFunctionDataset(Dataset):
    """
    Expects:
      - embedding_dir: *.pt tensors shaped [L, D]
      - go_mapping_file: TSV with: protein_id \t GO1;GO2;...
      - go_vocab_path (optional): If provided, fixes label order; otherwise will build from data.
      - ancestor propagation applied to labels.
    """
    def __init__(self, embedding_dir, go_mapping_file, go_graph,
                 go_vocab_path=None, build_vocab=False, max_len=None):
        self.embedding_dir = embedding_dir
        self.go_graph = go_graph
        self.max_len = max_len  # if not None, truncate/pad sequence len

        # files
        all_ids = [fn[:-3] for fn in os.listdir(embedding_dir) if fn.endswith(".pt")]
        self.ids = set(all_ids)

        # parse mapping and propagate
        self.go_labels = defaultdict(list)
        go_terms_set = set()
        with open(go_mapping_file, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2: 
                    continue
                pid, terms = parts
                if pid in self.ids:
                    term_list = [t.strip() for t in terms.split(";") if t.strip()]
                    full_terms = propagate_terms(term_list, go_graph)
                    self.go_labels[pid] = list(full_terms)
                    go_terms_set.update(full_terms)

        # Determine vocab (label order)
        if go_vocab_path and not build_vocab:
            with open(go_vocab_path, "r") as f:
                loaded = json.load(f)
            # support dict or list
            if isinstance(loaded, dict):
                self.go_vocab = loaded
            else:
                self.go_vocab = {go: i for i, go in enumerate(loaded)}
        else:
            # build from observed propagated labels
            self.go_vocab = {go_term: idx for idx, go_term in enumerate(sorted(go_terms_set))}

        self.num_labels = len(self.go_vocab)
        self.ids = list(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [L, D]
        if self.max_len is not None:
            L = emb.size(0)
            if L > self.max_len:
                emb = emb[:self.max_len]
            elif L < self.max_len:
                pad = torch.zeros(self.max_len - L, emb.size(1), dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
        attn_mask = (emb.sum(dim=1) != 0).long()
        y = torch.zeros(self.num_labels)
        for term in self.go_labels.get(pid, []):
            if term in self.go_vocab:
                y[self.go_vocab[term]] = 1.0
        return pid, emb, attn_mask, y

# ----------------------------
# Model
# ----------------------------
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
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ----------------------------
# Recall-friendly losses
# ----------------------------
class AsymmetricLoss(nn.Module):
    """
    Asymmetric (focal-like) loss to downweight easy negatives.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, eps=1e-8, label_smoothing=0.0):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.eps = eps
        self.ls = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        # standard BCE with logits (per-element)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # probs for modulating factor
        xs = torch.sigmoid(logits)
        pt = targets * xs + (1 - targets) * (1 - xs)
        gamma = self.gp * targets + self.gn * (1 - targets)
        loss *= (1 - pt).pow(gamma)
        return loss.mean()

def compute_label_freqs(dataset, num_labels):
    """
    One pass to estimate p(y=1) per class for logit adjustment or pos_weight.
    """
    counts = torch.zeros(num_labels, dtype=torch.float64)
    total = 0
    for _, _, _, y in tqdm(DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2),
                           desc="[INFO] Counting label freqs"):
        counts += y.sum(dim=0).double()
        total += y.size(0)
    # p = (#positives) / (#samples)
    freqs = (counts / max(total, 1)).clamp(min=1e-8, max=1.0 - 1e-8)
    return freqs.float()

def hierarchy_regularizer(logits, parent_mat, margin=0.0):
    """
    Enforce child <= parent (logit-wise) softly:
      penalty = ReLU(logit_child - logit_parent - margin)
    parent_mat: Sparse or dense [C, C] binary, parent_mat[child, parent]=1 if parent of child
    """
    # logits: [B, C]
    if parent_mat is None:
        return logits.new_tensor(0.0)
    # compute differences only where parent_mat=1
    # Efficient dense version for moderate C. For huge C, consider sparse ops.
    child_parent_diff = logits.unsqueeze(2) - logits.unsqueeze(1)  # [B, C, C] (child - parent)
    mask = parent_mat.unsqueeze(0).bool()  # [1, C, C]
    viol = torch.relu(child_parent_diff[mask] - margin)
    if viol.numel() == 0:
        return logits.new_tensor(0.0)
    return viol.mean()

# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo_path", default="/data/shared/databases/UniProt2025/GO_June_1_2025.obo")
    ap.add_argument("--embedding_dir", default="/data/archives/naufal/final_embeddings")
    ap.add_argument("--go_mapping_file", default="/data/summer2020/naufal/matched_ids_with_go.txt")
    ap.add_argument("--go_vocab_in", default="/data/shared/github/lutesAI2025/naufal/go_vocab.json")
    ap.add_argument("--go_vocab_out", default="/data/shared/github/lutesAI2025/naufal/go_vocab_full.json")
    ap.add_argument("--model_out", default="/data/shared/github/lutesAI2025/naufal/bigbird_finetuned_full.pt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--input_dim", type=int, default=1541)  # your final embedding size
    ap.add_argument("--max_len", type=int, default=1913)    # your fixed length
    ap.add_argument("--use_asl", action="store_true", help="Use Asymmetric Loss instead of BCE")
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    ap.add_argument("--gamma_pos", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_logit_adjust", action="store_true", help="Use logit adjustment with label priors")
    ap.add_argument("--use_pos_weight", action="store_true", help="Use BCE pos_weight from imbalance")
    ap.add_argument("--hier_margin", type=float, default=0.0, help="Hierarchy margin; 0.0 is standard child<=parent")
    ap.add_argument("--hier_weight", type=float, default=0.05, help="Weight for hierarchy regularizer")
    args = ap.parse_args()

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    print("[INFO] Parsing GO DAG + namespaces...")
    go_graph, go_ns = extract_go_graph_and_ns(args.obo_path)

    print("[INFO] Building dataset...")
    dataset = ProteinFunctionDataset(
        args.embedding_dir,
        args.go_mapping_file,
        go_graph,
        go_vocab_path=args.go_vocab_in if os.path.exists(args.go_vocab_in) else None,
        build_vocab=not os.path.exists(args.go_vocab_in),
        max_len=args.max_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    C = dataset.num_labels

    # Parent matrix for hierarchy regularizer
    print("[INFO] Building parent matrix for hierarchy regularization...")
    parent_mat = torch.zeros((C, C), dtype=torch.float32)
    id2go = {i: go for go, i in dataset.go_vocab.items()}
    for child_idx in range(C):
        child_go = id2go[child_idx]
        for par_go in go_graph.get(child_go, []):
            if par_go in dataset.go_vocab:
                parent_idx = dataset.go_vocab[par_go]
                parent_mat[child_idx, parent_idx] = 1.0
    parent_mat = parent_mat.to(device)

    # Compute priors if needed
    logit_adjust = None
    pos_weight = None
    if args.use_logit_adjust or args.use_pos_weight:
        print("[INFO] Estimating label frequencies...")
        freqs = compute_label_freqs(dataset, C)  # p(y=1)
        if args.use_logit_adjust:
            # adjustment ~ -log(prior): rarer labels get larger positive bias
            logit_adjust = (-torch.log(freqs)).to(device)  # [C]
        if args.use_pos_weight:
            # pos_weight ~ N_neg / N_pos = (1-p)/p
            pos_weight = ((1 - freqs) / freqs).clamp(1.0, 1e6).to(device)  # [C]

    # Model & optim
    model = BigBirdProteinModel(input_dim=args.input_dim, target_dim=C).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=args.min_lr)

    if args.use_asl:
        criterion = AsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
                                   label_smoothing=args.label_smoothing)
    else:
        # BCE with label smoothing + optional pos_weight
        class BCEWithLS(nn.Module):
            def __init__(self, ls=0.0, pos_weight_vec=None):
                super().__init__()
                self.ls = ls
                self.pos_weight = pos_weight_vec
            def forward(self, logits, targets):
                if self.ls > 0:
                    targets = targets * (1 - self.ls) + 0.5 * self.ls
                return F.binary_cross_entropy_with_logits(
                    logits, targets, weight=None, pos_weight=self.pos_weight, reduction="mean"
                )
        criterion = BCEWithLS(ls=args.label_smoothing, pos_weight_vec=pos_weight)

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader)):
            _, x, attn_mask, y = batch
            x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(x, attn_mask)
                # logit adjustment (bias for rare labels)
                if logit_adjust is not None:
                    logits = logits + logit_adjust.unsqueeze(0)  # broadcast [1, C]
                base_loss = criterion(logits, y)
                reg = hierarchy_regularizer(logits, parent_mat, margin=args.hier_margin) * args.hier_weight
                loss = base_loss + reg

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i == 0 or (i + 1) % 500 == 0:
                print(f"[✓] Trained {i + 1:,} proteins; batch loss={loss.item():.5f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.6f} (base={base_loss.item():.6f}, reg={reg.item():.6f})")
        scheduler.step(avg_loss)

        # quick sanity peek
        model.eval()
        with torch.no_grad():
            pid0, sx, smask, _ = dataset[0]
            sx, smask = sx.unsqueeze(0).to(device), smask.unsqueeze(0).to(device)
            sout = torch.sigmoid(model(sx, smask))
            print("[DEBUG] Sample probs (first 10):", sout.squeeze()[:10].detach().cpu().numpy())
        model.train()

    torch.save(model.state_dict(), args.model_out)
    print(f"[✓] Model saved to {args.model_out}")

    # Save the label order used for training
    with open(args.go_vocab_out, "w") as f:
        json.dump(dataset.go_vocab, f)
    print(f"[✓] Saved GO vocabulary to {args.go_vocab_out}")


if __name__ == "__main__":
    main()
