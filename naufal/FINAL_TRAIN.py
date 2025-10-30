#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel, BigBirdConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------
# GO utilities
# ----------------------------
def extract_go_graph_and_ns(obo_path):
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
                go_namespace[current_id] = line.split("namespace: ")[1]
    return go_graph, go_namespace

def propagate_terms(go_terms, go_graph):
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
    def __init__(self, embedding_dir, go_mapping_file, go_graph,
                 go_vocab_path=None, build_vocab=False, max_len=None):
        self.embedding_dir = embedding_dir
        self.go_graph = go_graph
        self.max_len = max_len

        ids = [fn[:-3] for fn in os.listdir(embedding_dir) if fn.endswith(".pt")]
        self.ids = set(ids)

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

        if go_vocab_path and not build_vocab and os.path.exists(go_vocab_path):
            with open(go_vocab_path, "r") as f:
                loaded = json.load(f)
            self.go_vocab = loaded if isinstance(loaded, dict) else {go: i for i, go in enumerate(loaded)}
        else:
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
            idx = self.go_vocab.get(term, None)
            if idx is not None:
                y[idx] = 1.0
        return pid, emb, attn_mask, y

# ----------------------------
# Model
# ----------------------------
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, gradient_checkpointing=True):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)

        # Disable the cache (saves memory) and enable gradient ckpt
        self.bigbird = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base",
            attention_type="block_sparse",
        )
        self.bigbird.config.use_cache = False
        if gradient_checkpointing:
            self.bigbird.gradient_checkpointing_enable()

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

def freeze_early_layers(model, n_layers_to_freeze=0):
    """
    Freeze embeddings and first N transformer blocks to save memory.
    """
    if n_layers_to_freeze <= 0:
        return
    # embeddings
    for p in model.bigbird.embeddings.parameters():
        p.requires_grad = False
    # first N layers
    layers = model.bigbird.encoder.layer
    for i in range(min(n_layers_to_freeze, len(layers))):
        for p in layers[i].parameters():
            p.requires_grad = False

# ----------------------------
# Recall-friendly losses (same APIs)
# ----------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, eps=1e-8, label_smoothing=0.0):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.eps = eps
        self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        xs = torch.sigmoid(logits)
        pt = targets * xs + (1 - targets) * (1 - xs)
        gamma = self.gp * targets + self.gn * (1 - targets)
        loss *= (1 - pt).pow(gamma)
        return loss.mean()

def compute_label_freqs(dataset, num_labels):
    counts = torch.zeros(num_labels, dtype=torch.float64)
    total = 0
    dl = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2,
                    pin_memory=True, persistent_workers=True)
    for _, _, _, y in tqdm(dl, desc="[INFO] Counting label freqs"):
        counts += y.sum(dim=0).double()
        total += y.size(0)
    freqs = (counts / max(total, 1)).clamp(min=1e-8, max=1.0 - 1e-8)
    return freqs.float()

def build_hierarchy_edges(go_graph, go_vocab):
    """
    Return a list of (child_idx, parent_idx) edges for sparse regularization.
    """
    edges = []
    for child_go, parents in go_graph.items():
        if child_go not in go_vocab:
            continue
        c = go_vocab[child_go]
        for pg in parents:
            if pg in go_vocab:
                edges.append((c, go_vocab[pg]))
    return edges

def hierarchy_regularizer_sparse(logits, edges, margin=0.0):
    """
    Sum over max(0, logit_child - logit_parent - margin) using sparse edges.
    logits: [B, C]; edges: list[(c,p)]
    """
    if not edges:
        return logits.new_tensor(0.0)
    diffs = []
    for (c, p) in edges:
        diffs.append((logits[:, c] - logits[:, p] - margin).relu())
    return torch.stack(diffs, dim=0).mean()

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

    ap.add_argument("--batch_size", type=int, default=2)          # small to avoid OOM
    ap.add_argument("--accum_steps", type=int, default=8)         # gradient accumulation
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--input_dim", type=int, default=1541)
    ap.add_argument("--max_len", type=int, default=1913)
    ap.add_argument("--fp16_inputs", action="store_true")         # cast embeddings to float16
    ap.add_argument("--freeze_layers", type=int, default=0)       # freeze early encoder blocks to save mem

    ap.add_argument("--use_asl", action="store_true")
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    ap.add_argument("--gamma_pos", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_logit_adjust", action="store_true")
    ap.add_argument("--use_pos_weight", action="store_true")
    ap.add_argument("--hier_margin", type=float, default=0.0)
    ap.add_argument("--hier_weight", type=float, default=0.05)

    args = ap.parse_args()

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    print("[INFO] Parse GO DAG + namespaces...")
    go_graph, go_ns = extract_go_graph_and_ns(args.obo_path)

    print("[INFO] Build dataset...")
    dataset = ProteinFunctionDataset(
        args.embedding_dir,
        args.go_mapping_file,
        go_graph,
        go_vocab_path=args.go_vocab_in,
        build_vocab=not os.path.exists(args.go_vocab_in),
        max_len=args.max_len
    )
    C = dataset.num_labels

    # DataLoader – keep small batches; workers persistent reduces loader overhead
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # helps with accumulation/bookkeeping
    )

    # Hierarchy as sparse edge list (saves tons of mem vs dense matrix)
    edges = build_hierarchy_edges(go_graph, dataset.go_vocab)

    # Priors for logit adjustment / pos_weight
    logit_adjust = None
    pos_weight = None
    if args.use_logit_adjust or args.use_pos_weight:
        print("[INFO] Estimating label frequencies (priors)...")
        freqs = compute_label_freqs(dataset, C)  # p(y=1)
        if args.use_logit_adjust:
            logit_adjust = (-torch.log(freqs)).to(device)  # rarer => larger positive bias
        if args.use_pos_weight:
            pos_weight = ((1 - freqs) / freqs).clamp(1.0, 1e6).to(device)

    # Model
    model = BigBirdProteinModel(input_dim=args.input_dim, target_dim=C, gradient_checkpointing=True).to(device)
    freeze_early_layers(model, n_layers_to_freeze=args.freeze_layers)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=args.min_lr)

    if args.use_asl:
        criterion = AsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
                                   label_smoothing=args.label_smoothing)
    else:
        class BCEWithLS(nn.Module):
            def __init__(self, ls=0.0, pos_weight_vec=None):
                super().__init__()
                self.ls = ls
                self.pos_weight = pos_weight_vec
            def forward(self, logits, targets):
                tt = targets
                if self.ls > 0:
                    tt = targets * (1 - self.ls) + 0.5 * self.ls
                return F.binary_cross_entropy_with_logits(
                    logits, tt, pos_weight=self.pos_weight, reduction="mean"
                )
        criterion = BCEWithLS(ls=args.label_smoothing, pos_weight_vec=pos_weight)

    scaler = torch.cuda.amp.GradScaler()

    # ---- Training loop with accumulation ----
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (_, x, attn_mask, y) in enumerate(tqdm(dataloader)):
            # move to device with minimal copies; optionally fp16 for inputs
            if args.fp16_inputs:
                x = x.half()
            x = x.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(x, attn_mask)
                if logit_adjust is not None:
                    logits = logits + logit_adjust.unsqueeze(0)
                base_loss = criterion(logits, y)
                reg = hierarchy_regularizer_sparse(logits, edges, margin=args.hier_margin) * args.hier_weight
                loss = (base_loss + reg) / args.accum_steps  # scale for accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item() * args.accum_steps  # undo scaling for logging

            if i == 0 or (i + 1) % (500 // max(args.batch_size,1)) == 0:
                print(f"[✓] Seen {i + 1:,} batches  |  batch_loss={base_loss.item():.5f}  reg={reg.item():.5f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

        # quick peek
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, sx, sm, _ = dataset[0]
            if args.fp16_inputs:
                sx = sx.half()
            sout = torch.sigmoid(model(sx.unsqueeze(0).to(device), sm.unsqueeze(0).to(device)))
            print("[DEBUG] Sample probs (first 10):", sout.squeeze()[:10].float().cpu().numpy())
        model.train()

        # try to keep GPU clean between epochs
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), args.model_out)
    print(f"[✓] Model saved to {args.model_out}")

    with open(args.go_vocab_out, "w") as f:
        json.dump(dataset.go_vocab, f)
    print(f"[✓] Saved GO vocabulary to {args.go_vocab_out}")

if __name__ == "__main__":
    main()

