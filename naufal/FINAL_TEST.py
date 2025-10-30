#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel
from tqdm import tqdm

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

def add_ancestors(indices, id2go, go2id, go_graph):
    """Add ancestors to a set of predicted indices (ontology closure)."""
    keep = set(indices)
    agenda = list(indices)
    while agenda:
        c = agenda.pop()
        go = id2go[c]
        for parent in go_graph.get(go, []):
            if parent in go2id:
                pid = go2id[parent]
                if pid not in keep:
                    keep.add(pid)
                    agenda.append(pid)
    return sorted(keep)

# ----------------------------
# Dataset
# ----------------------------
class TestProteinDataset(Dataset):
    def __init__(self, embedding_dir, max_len=None):
        self.embedding_dir = embedding_dir
        self.ids = [fn[:-3] for fn in os.listdir(embedding_dir) if fn.endswith(".pt")]
        self.max_len = max_len

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
        attn_mask = (emb.sum(dim=1) != 0).long()
        return pid, emb, attn_mask

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
        out = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ----------------------------
# Decoding
# ----------------------------
def decode_predictions(
    probs_row,
    taus=None,
    topk=50,
    protein_percentile=None,  # e.g., 0.92 means keep scores >= 92nd percentile
    bias_logit_delta=0.0,     # e.g., +0.2 to be more generous
    temperature=1.0,          # <1.0 sharpens, >1.0 smooths
):
    """
    Returns a set of predicted indices BEFORE ontology closure.
    """
    # optional bias at logit-level approx by inverse-sigmoid
    if bias_logit_delta != 0.0 or temperature != 1.0:
        # convert to logits, apply bias/temperature, back to probs
        eps = 1e-8
        p = probs_row.clip(eps, 1 - eps)
        logits = np.log(p/(1-p))
        logits = (logits + bias_logit_delta) / max(temperature, 1e-6)
        probs_row = 1.0 / (1.0 + np.exp(-logits))

    C = probs_row.shape[0]
    picked = set()

    # 1) Per-label thresholds
    if taus is not None:
        above = np.where(probs_row >= taus)[0]
        picked.update(above.tolist())

    # 2) Protein-adaptive percentile threshold (optional)
    if protein_percentile is not None:
        thr = np.quantile(probs_row, protein_percentile)
        picked.update(np.where(probs_row >= thr)[0].tolist())

    # 3) Ensure at least top-K predictions
    if len(picked) < topk:
        topk_idx = np.argsort(-probs_row)[:topk]
        picked.update(topk_idx.tolist())

    return sorted(picked)

def apply_namespace_caps(pred_indices, id2go, go_ns, caps=None):
    """
    caps: dict like {'biological_process': 30, 'molecular_function': 10, 'cellular_component': 10}
    If None, skip.
    """
    if not caps:
        return pred_indices
    bucket = defaultdict(list)
    for idx in pred_indices:
        go = id2go[idx]
        ns = go_ns.get(go, "unknown")
        bucket[ns].append(idx)
    kept = []
    for ns, inds in bucket.items():
        k = caps.get(ns, len(inds))
        kept.extend(inds[:k])  # inds already in score-desc order before calling? ensure we sort externally
    # Keep original order as best as possible
    return kept

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dir", default="/data/summer2020/naufal/testing_pca")
    ap.add_argument("--vocab_path", default="/data/shared/github/lutesAI2025/naufal/go_vocab.json")
    ap.add_argument("--model_path", default="/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt")
    ap.add_argument("--obo_path", default="/data/shared/databases/UniProt2025/GO_June_1_2025.obo")
    ap.add_argument("--thresholds_path", default="thresholds.npy", help="Per-label τ_j; optional")
    ap.add_argument("--output_file", default="testing_predictions.txt")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--input_dim", type=int, default=1541)  # set to 512 if your test embeddings are PCA=512
    ap.add_argument("--max_len", type=int, default=1913)
    # decoding knobs
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--protein_percentile", type=float, default=None)  # e.g., 0.92
    ap.add_argument("--bias_logit_delta", type=float, default=0.2)     # positive makes model more generous
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use_ns_caps", action="store_true")
    ap.add_argument("--bp_cap", type=int, default=30)
    ap.add_argument("--mf_cap", type=int, default=10)
    ap.add_argument("--cc_cap", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    print("[INFO] Loading GO vocabulary...")
    with open(args.vocab_path, "r") as f:
        vocab = json.load(f)
    go2id = vocab if isinstance(vocab, dict) else {go: i for i, go in enumerate(vocab)}
    id2go = {i: go for go, i in go2id.items()}
    C = len(go2id)

    print("[INFO] Parsing GO DAG + namespaces...")
    go_graph, go_ns = extract_go_graph_and_ns(args.obo_path)

    # thresholds (optional)
    taus = None
    if os.path.exists(args.thresholds_path):
        taus = np.load(args.thresholds_path)
        if taus.shape[0] != C:
            raise ValueError(f"thresholds.npy has C={taus.shape[0]} but vocab has C={C}")

    print("[INFO] Loading test dataset...")
    dataset = TestProteinDataset(args.embedding_dir, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model
    print("[INFO] Loading model...")
    model = BigBirdProteinModel(args.input_dim, C).to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # namespace caps
    caps = None
    if args.use_ns_caps:
        caps = {
            "biological_process": args.bp_cap,
            "molecular_function": args.mf_cap,
            "cellular_component": args.cc_cap,
        }

    print("[INFO] Running inference...")
    with open(args.output_file, "w") as out_f, torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            pids, x, attn_mask = batch
            x, attn_mask = x.to(device), attn_mask.to(device)
            logits = model(x, attn_mask)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]

            # pre-sort indices for NS caps by score
            order = np.argsort(-probs, axis=1)

            for b in range(probs.shape[0]):
                prob_row = probs[b]
                # decode generously
                pred_idx = decode_predictions(
                    prob_row,
                    taus=taus,
                    topk=args.topk,
                    protein_percentile=args.protein_percentile,
                    bias_logit_delta=args.bias_logit_delta,
                    temperature=args.temperature,
                )

                # reorder pred_idx by score desc (needed before caps)
                pred_idx = sorted(pred_idx, key=lambda j: -prob_row[j])

                # optional namespace caps
                if caps:
                    pred_idx = apply_namespace_caps(pred_idx, id2go, go_ns, caps=caps)

                # ontology closure (add ancestors)
                pred_idx = add_ancestors(pred_idx, id2go, go2id, go_graph)

                terms = [id2go[j] for j in pred_idx]
                out_f.write(f"{pids[b]}\t{';'.join(terms)}\n")

            if i == 0 or (i + 1) % 500 == 0:
                print(f"[✓] Predicted {i + 1:,} proteins")

    print(f"[✓] Saved predictions to {args.output_file}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
