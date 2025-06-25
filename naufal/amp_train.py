import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BigBirdModel
from tqdm import tqdm
from wrap_data import ProteinFunctionDataset
from sklearn.decomposition import PCA

# === Custom PCA Wrapper ===
class TokenPCA:
    def __init__(self, original_tokens=1913, target_tokens=510):
        self.original_tokens = original_tokens
        self.target_tokens = target_tokens
        self.pca = PCA(n_components=target_tokens)

    def fit(self, sample_tensor_batch):
        """
        sample_tensor_batch: shape (N, T, D) → (N×D, T)
        We'll fit across tokens (T) using flattened batches.
        """
        N, T, D = sample_tensor_batch.shape
        assert T == self.original_tokens
        flattened = sample_tensor_batch.permute(0, 2, 1).reshape(-1, T)  # (N×D, T)
        self.pca.fit(flattened)

    def transform(self, embeddings):
        """
        embeddings: shape (B, T, D) → return shape (B, T_reduced, D)
        """
        B, T, D = embeddings.shape
        flattened = embeddings.permute(0, 2, 1).reshape(-1, T)  # (B×D, T)
        reduced = self.pca.transform(flattened)  # (B×D, T_reduced)
        output = torch.tensor(reduced, dtype=embeddings.dtype, device=embeddings.device)
        return output.reshape(B, D, -1).permute(0, 2, 1)  # (B, T_reduced, D)

# === Custom BigBird Model ===
class CustomBigBirdModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(input_dim, 768)
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base')
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)
        self.bigbird.gradient_checkpointing_enable()

    def forward(self, embeddings, attention_mask):
        projected = self.embedding_proj(embeddings)
        outputs = self.bigbird(inputs_embeds=projected, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# === Training Function ===
def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs, device, save_dir, pca):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for embeddings, labels, _ in progress:
            embeddings, labels = embeddings.to(device), labels.to(device)

            # === Apply PCA to reduce token length ===
            embeddings = pca.transform(embeddings)  # (B, 510, 1541)

            attention_mask = (embeddings.abs().sum(dim=-1) != 0).int()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(embeddings, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(total_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[LR] Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

        print(f"[✓] Epoch {epoch+1} complete | Loss: {total_loss:.4f}")
        save_path = os.path.join(save_dir, f"proteinext_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Model saved to {save_path}")

    return model

# === Main ===
if __name__ == "__main__":
    EMBEDDING_DIR = "/data/archives/naufal/final_embeddings"
    GO_MAPPING_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"
    SAVE_DIR = "/data/summer2020/naufal/proteinext_bigbird_checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = ProteinFunctionDataset(EMBEDDING_DIR, GO_MAPPING_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = CustomBigBirdModel(input_dim=1541, num_classes=dataset.num_labels)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # === Fit PCA on a subset of the dataset (e.g., 512 samples) ===
    print("[✓] Fitting PCA on token dimension...")
    sample_batch = []
    for i, (emb, _, _) in enumerate(dataloader):
        sample_batch.append(emb)
        if i >= 15: break  # ~480 samples
    sample_tensor = torch.cat(sample_batch, dim=0)  # (N, 1913, 1541)

    pca = TokenPCA(original_tokens=1913, target_tokens=510)
    pca.fit(sample_tensor)

    # === Begin Training ===
    print("[✓] Starting training with PCA-reduced token embeddings...")
    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=5,
        device=device,
        save_dir=SAVE_DIR,
        pca=pca
    )

    print("[✓] Training complete.")

