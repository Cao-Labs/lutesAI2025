import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BigBirdModel, BigBirdConfig
from collections import defaultdict
from tqdm import tqdm

# === Dataset Class ===
class ProteinTestDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir
        self.ids = [fname[:-3] for fname in os.listdir(embedding_dir) if fname.endswith(".pt")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        embedding = torch.load(os.path.join(self.embedding_dir, f"{pid}.pt"))  # [512, 512]
        attention_mask = (embedding.sum(dim=1) != 0).long()  # [512]
        return pid, embedding, attention_mask

# === Model Definition ===
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, max_len):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)
        config = BigBirdConfig(
            vocab_size=50265,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            attention_type="block_sparse",
            block_size=64,
            max_position_embeddings=max_len,
            use_bias=True,
            is_decoder=False,
        )
        self.bigbird = BigBirdModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)  # [B, L, 768]
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(cls_output)  # [B, num_labels]
        return logits

# === Load GO Vocabulary from Training File ===
def load_go_vocab(go_mapping_file):
    go_terms_set = set()
    with open(go_mapping_file, "r") as f:
        for line in f:
            _, terms = line.strip().split("\t")
            term_list = terms.split(";")
            go_terms_set.update(term_list)
    go_vocab = {idx: go_term for idx, go_term in enumerate(sorted(go_terms_set))}
    return go_vocab

# === Prediction Function ===
def predict():
    # === Config ===
    embedding_dir = "/data/summer2020/naufal/testing_pca"
    go_mapping_file = "/data/summer2020/naufal/matched_ids_with_go.txt"  # TRAINING MAPPING
    model_path = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt"
    output_file = "/data/summer2020/naufal/test_pred.txt"
    batch_size = 1
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    # === Load GO vocab ===
    go_vocab = load_go_vocab(go_mapping_file)
    idx_to_go = go_vocab
    num_labels = len(idx_to_go)

    # === Load model ===
    model = BigBirdProteinModel(input_dim=512, target_dim=num_labels, max_len=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Load test data ===
    dataset = ProteinTestDataset(embedding_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with open(output_file, "w") as out_f:
        for i, (pid, x, attn_mask) in enumerate(tqdm(dataloader)):
            x, attn_mask = x.to(device), attn_mask.to(device)
            with torch.no_grad():
                logits = model(x, attn_mask)
                preds = torch.sigmoid(logits) > 0.5

            pred_go_terms = [idx_to_go[idx] for idx in torch.where(preds[0])[0].tolist()]
            out_f.write(f"{pid[0]}\t{';'.join(pred_go_terms)}\n")

            if i == 0 or (i + 1) % 1000 == 0:
                print(f"[✓] Predicted {i + 1:,} proteins")

    print(f"[✓] Predictions written to: {output_file}")

if __name__ == "__main__":
    predict()
