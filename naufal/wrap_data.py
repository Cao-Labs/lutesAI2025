import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# === Custom Dataset ===
class ProteinFunctionDataset(Dataset):
    def __init__(self, embedding_dir, go_mapping_file):
        self.embedding_dir = embedding_dir
        self.go_mapping_file = go_mapping_file

        # Step 1: Get all .pt embedding IDs
        print("[INFO] Scanning embedding files...")
        self.ids = set(fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt"))
        print(f"[INFO] Found {len(self.ids):,} embedding files.")

        # Step 2: Parse GO terms for matching IDs
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

        return embedding, target, pid

# === Main Block ===
if __name__ == "__main__":
    EMBEDDING_DIR = "/data/archives/naufal/final_embeddings"
    GO_MAPPING_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"

    # Create the dataset
    dataset = ProteinFunctionDataset(EMBEDDING_DIR, GO_MAPPING_FILE)

    # Wrap in DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,       # You can increase this if CPU allows
        pin_memory=True      # Speeds up GPU training
    )

    # Peek at the first batch
    for i, (embeddings, targets, ids) in enumerate(dataloader):
        print(f"[✓] Loaded batch {i + 1}")
        print(f" - Embedding shape: {embeddings.shape}")
        print(f" - Target shape: {targets.shape}")
        print(f" - First protein ID: {ids[0]}")
        break
