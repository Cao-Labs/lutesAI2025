import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ProteinFunctionDataset(Dataset):
    def __init__(self, embedding_dir, go_mapping_file):
        self.embedding_dir = embedding_dir
        self.go_mapping_file = go_mapping_file

        # Step 1: Scan .pt embedding files (use a set for fast lookup)
        print("[INFO] Scanning embedding files...")
        self.ids = set(fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt"))
        print(f"[INFO] Found {len(self.ids):,} embedding files.")

        # Step 2: Build GO label dictionary for relevant IDs
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

        # Step 3: Build GO vocabulary
        self.go_vocab = {go_term: idx for idx, go_term in enumerate(sorted(go_terms_set))}
        self.num_labels = len(self.go_vocab)
        print(f"[INFO] Constructed GO vocabulary with {self.num_labels} terms.")

        # Convert set of ids back to list for indexing
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

        if idx == 0 or idx % 10000 == 0:
            print(f"[✓] Processed protein {idx + 1:,}: {pid} | shape: {embedding.shape} | GO terms: {int(target.sum().item())}")

        return embedding, target, pid


# === TESTING BLOCK ===
if __name__ == "__main__":
    dataset = ProteinFunctionDataset(
        embedding_dir="/data/archives/naufal/final_embeddings",
        go_mapping_file="/data/summer2020/naufal/matched_ids_with_go.txt"
    )

    print(f"[INFO] Total proteins in dataset: {len(dataset):,}")

    # Iterate to trigger progress printing for every 10,000
    for idx in range(len(dataset)):
        _ = dataset[idx]

