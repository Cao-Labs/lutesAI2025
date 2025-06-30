import os
import json
import torch
import torch.nn as nn
from transformers import BigBirdModel
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

# === Load GO vocab and labels ===
print("Loading GO vocab...")
with open(GO_VOCAB_PATH) as f:
    go_vocab = json.load(f)

go_classes = sorted(go_vocab.keys())
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
for fname in os.listdir(EMBEDDING_DIR):
    if fname.endswith(".pt"):
        pid = fname[:-3]
        if pid in labels_dict:
            emb = torch.load(os.path.join(EMBEDDING_DIR, fname))  # [512, 512]
            embeddings.append(emb)
            label_list.append(labels_dict[pid])
            ids.append(pid)

# === Convert to tensors ===
print("Encoding labels...")
label_tensor = torch.tensor(mlb.fit_transform(label_list), dtype=torch.float32)
embedding_tensor = torch.stack(embeddings).float()

dataset = TensorDataset(embedding_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Define Model ===
class CustomBigBirdModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding_proj = nn.Linear(EMBEDDING_DIM, 768)
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base', ignore_mismatched_sizes=True)
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding_proj(x)
        x = x.unsqueeze(1)  # [B, 1, 768]
        output = self.bigbird(inputs_embeds=x)
        cls_output = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# === Load model ===
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
num_classes = len(mlb.classes_)
model = CustomBigBirdModel(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Run inference ===
print("Running inference...")
all_preds, all_true = [], []
with torch.no_grad():
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu())
        all_true.append(y.cpu())

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

with open(PRED_OUTPUT, "w") as f:
    f.write("\n".join(pred_lines))

# === Write actual ground truth in CAFA format ===
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

with open(ACTUAL_LABEL_PATH.replace(".txt", "_actual_cafa.txt"), "w") as f:
    f.write("\n".join(actual_lines))

print("Done.")


