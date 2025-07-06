import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel
from tqdm import tqdm

# === Dataset for Testing (no labels needed) ===
class TestProteinDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        self.ids = [fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [1913, 1541]
        attention_mask = (embedding.sum(dim=1) != 0).long()  # [1913]
        return pid, embedding, attention_mask

# === Model Definition (must match training) ===
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)
        self.bigbird = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base",
            attention_type="block_sparse"
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# === Inference Script ===
def test():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    batch_size = 16
    threshold = 0.69

    # === Paths ===
    embedding_dir = "/data/summer2020/naufal/testing_normalized"
    vocab_path = "/data/shared/github/lutesAI2025/naufal/go_vocab_full.json"
    model_path = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned_full.pt"
    output_file = "/data/summer2020/naufal/full_pred.txt"

    print("[INFO] Loading GO vocabulary...")
    with open(vocab_path, "r") as f:
        go_vocab = json.load(f)
    id2go = {idx: go for go, idx in go_vocab.items()}
    num_labels = len(go_vocab)

    print("[INFO] Loading test dataset...")
    dataset = TestProteinDataset(embedding_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("[INFO] Loading model...")
    model = BigBirdProteinModel(input_dim=1541, target_dim=num_labels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[INFO] Running inference...")
    with open(output_file, "w") as out_f:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                pids, x, attn_mask = batch
                x, attn_mask = x.to(device), attn_mask.to(device)
                logits = model(x, attn_mask)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

                for pid, pred_row in zip(pids, preds):
                    terms = [id2go[idx] for idx, val in enumerate(pred_row) if val == 1]
                    out_f.write(f"{pid}\t{';'.join(terms)}\n")

                if i == 0 or (i + 1) % 500 == 0:
                    print(f"[✓] Predicted {i + 1:,} proteins")

    print(f"[✓] Saved predictions to {output_file}")

# === Entry Point ===
if __name__ == "__main__":
    test()
