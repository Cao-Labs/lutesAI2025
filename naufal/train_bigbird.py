import os
import torch
from torch.utils.data import DataLoader
from transformers import BigBirdConfig, BigBirdForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from wrap_data import ProteinFunctionDataset  # Make sure wrap_data.py is in the same dir or installed as module

# === Config ===
EMBEDDING_DIR = "/data/archives/naufal/final_embeddings"
GO_MAPPING_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-5
MIN_LR = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load dataset and dataloader ===
print("[INFO] Loading dataset...")
dataset = ProteinFunctionDataset(EMBEDDING_DIR, GO_MAPPING_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# === BigBird model setup ===
print("[INFO] Initializing BigBird model...")
config = BigBirdConfig(
    vocab_size=50265,
    attention_type="block_sparse",
    max_position_embeddings=1913,
    num_labels=dataset.num_labels,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12
)
model = BigBirdForSequenceClassification(config)
model.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
model.to(DEVICE)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
    "reduce_on_plateau",
    optimizer=optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=MIN_LR
)

loss_fn = BCEWithLogitsLoss()

# === Training Loop ===
print("[INFO] Starting training...")
global_step = 0
for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    model.train()
    epoch_loss = 0.0

    for i, (embeddings, targets, ids) in enumerate(tqdm(dataloader)):
        embeddings = embeddings.to(DEVICE)      # [B, L, D]
        targets = targets.to(DEVICE)

        # Attention mask: 1 where sequence is non-zero
        attention_mask = (embeddings.abs().sum(dim=-1) != 0).long()

        # Reduce D to match BigBird hidden size
        projected = torch.nn.functional.linear(embeddings, torch.randn(768, embeddings.shape[2]).to(DEVICE))

        outputs = model(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        global_step += 1

        if i == 0 or global_step % 10000 < BATCH_SIZE:
            print(f"[✓] Trained {global_step:,} proteins | Batch {i+1} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Average loss: {avg_loss:.4f}")
    lr_scheduler.step(avg_loss)

print("[✓] Training complete.")
