import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BigBirdModel
from tqdm import tqdm
from wrap_data import ProteinFunctionDataset  # your dataset script

# === Custom BigBird Model for Protein Function Prediction ===
class CustomBigBirdModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(input_dim, 768)  # Project to BigBird input dim
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base')
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings, attention_mask):
        projected = self.embedding_proj(embeddings)  # (B, L, 768)
        outputs = self.bigbird(inputs_embeds=projected, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits

# === Training Function ===
def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs, device, save_dir):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for embeddings, labels, _ in progress:
            embeddings, labels = embeddings.to(device), labels.to(device)

            # Create attention mask: 1 where embedding is non-zero, else 0
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

            torch.cuda.empty_cache()  # Clear cache to reduce OOM risk

        # Step LR scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(total_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[LR] Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

        print(f"[\u2713] Epoch {epoch+1} complete | Loss: {total_loss:.4f}")

        # Save model
        save_path = os.path.join(save_dir, f"proteinext_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[\u2713] Model saved to {save_path}")

    return model

# === Main Block ===
if __name__ == "__main__":
    # Filepaths
    EMBEDDING_DIR = "/data/archives/naufal/final_embeddings"
    GO_MAPPING_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"
    SAVE_DIR = "/data/summer2020/naufal/proteinext_bigbird_checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Dataset & Dataloader
    dataset = ProteinFunctionDataset(EMBEDDING_DIR, GO_MAPPING_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Lowered batch size for memory efficiency
        shuffle=True,
        num_workers=2,  # Tune based on system capacity
        pin_memory=True
    )

    # Model and training setup
    input_dim = 1541
    num_classes = dataset.num_labels
    model = CustomBigBirdModel(input_dim=input_dim, num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
    )
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[\u2713] Starting training with AMP and batch size 4...")
    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=5,
        device=device,
        save_dir=SAVE_DIR
    )

    print("[\u2713] Training complete.")

