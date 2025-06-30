import os
import json
import torch
import torch.nn as nn
from transformers import BigBirdModel, BigBirdConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer

# === Paths ===
EMBEDDING_DIR = "/data/summer2020/naufal/testing_pca"
GO_VOCAB_PATH = "/data/shared/github/lutesAI2025/naufal/go_vocab.json"
MODEL_PATH = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned.pt"
PRED_OUTPUT = "/data/summer2020/naufal/test_pred.txt"
ACTUAL_LABEL_PATH = "/data/summer2020/naufal/matched_ids_with_go.txt"

# === Constants ===
BATCH_SIZE = 8
THRESHOLD = 0.69
EMBEDDING_DIM = 512
MAX_POS = 512

# === Load GO vocab and labels ===
print("Loading GO vocab...")
with open(GO_VOCAB_PATH) as f:
    go_vocab = json.load(f)

go_classes = sorted(go_vocab.keys())
vocab_size = len(go_vocab)
mlb = MultiLabelBinarizer(classes=go_classes)

print("Loading ground truth labels...")
labels_dict = {}
with open(ACTUAL_LABEL_PATH, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, go_terms_str = parts
        go_terms = [t.strip() for t in go_terms_str.split(",") if t.strip() in go_classes]
        labels_dict[pid] = go_terms

# === Gather matching .pt embeddings and labels ===
embeddings, label_list, ids = [], [], []
print("Loading embeddings and matching labels...")
count = 0
for fname in sorted(os.listdir(EMBEDDING_DIR)):
    if fname.endswith(".pt"):
        pid = fname[:-3]
        if pid in labels_dict:
            emb = torch.load(os.path.join(EMBEDDING_DIR, fname))  # [512, 512]
            embeddings.append(emb)
            label_list.append(labels_dict[pid])
            ids.append(pid)
            count += 1
            if count % 500 == 0:
                print(f" → Loaded {count} protein embeddings...")

print(f"✅ Total proteins loaded: {count}")

# === Convert to tensors ===
print("Encoding labels...")
label_tensor = torch.tensor(mlb.fit_transform(label_list), dtype=torch.float32)
embedding_tensor = torch.stack(embeddings).float()

dataset = TensorDataset(embedding_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Match model to checkpoint ===
class BigBirdProteinModel(nn.Module):
    def __init__(self, input_dim, target_dim, max_len, vocab_size):
        super().__init__()
        self.project = nn.Linear(input_dim, 768)
        config = BigBirdConfig(
            vocab_size=vocab_size,
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
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, target_dim)
        )

    def forward(self, x, attention_mask):
        x = self.project(x)
        outputs = self.bigbird(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# === Load model ===
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
num_classes = len(mlb.classes_)
model = BigBirdProteinModel(input_dim=EMBEDDING_DIM, target_dim=num_classes, max_len=MAX_POS, vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Run inference ===
print("Running inference...")
all_preds, all_true = [], []
processed = 0
with torch.no_grad():
    for x_batch, y_batch in dataloader:
        attn_mask = (x_batch.sum(dim=2) != 0).long()
        x_batch, y_batch, attn_mask = x_batch.to(device), y_batch.to(device), attn_mask.to(device)
        logits = model(x_batch, attn_mask)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu())
        all_true.append(y_batch.cpu())

        processed += x_batch.size(0)
        if processed % 500 == 0:
            print(f" → Inference complete for {processed} proteins...")

print("✅ Inference complete.")

all_preds = torch.cat(all_preds)
all_true = torch.cat(all_true)
pred_binary = (all_preds > THRESHOLD).float()
accuracy_per_label = (pred_binary == all_true).float().mean(dim=0)

# === Write predictions in CAFA format ===
print("Writing predictions...")
pred_lines = [
    "AUTHOR\tAlphaAnalyzers",
    "MODEL\tProteinext",
    "KEYWORDS\tde novo prediction, machine learning."
]

for i, pid in enumerate(ids):
    for j, go_term in enumerate(mlb.classes_):
        if pred_binary[i, j] == 1:
            acc = round(accuracy_per_label[j].item(), 2)
            pred_lines.append(f"{pid}\t{go_term}\t{acc}")
    if (i + 1) % 500 == 0:
        print(f" → Predictions written for {i+1} proteins...")

with open(PRED_OUTPUT, "w") as f:
    f.write("\n".join(pred_lines))

# === Write ground truth in CAFA format ===
print("Writing ground truth...")
actual_lines = [
    "AUTHOR\tAlphaAnalyzers",
    "MODEL\tProteinext",
    "KEYWORDS\tde novo prediction, machine learning."
]

for i, pid in enumerate(ids):
    for j, go_term in enumerate(mlb.classes_):
        if all_true[i, j] == 1:
            actual_lines.append(f"{pid}\t{go_term}")
    if (i + 1) % 500 == 0:
        print(f" → Ground truth written for {i+1} proteins...")

gt_path = ACTUAL_LABEL_PATH.replace(".txt", "_actual_cafa.txt")
with open(gt_path, "w") as f:
    f.write("\n".join(actual_lines))

print("✅ Evaluation complete.")


