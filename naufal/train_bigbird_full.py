# === Training Loop ===
def train():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    batch_size = 16
    epochs = 5
    learning_rate = 1e-5
    min_lr = 1e-7

    # === Paths ===
    obo_path = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
    embedding_dir = "/data/archives/naufal/final_embeddings"
    go_mapping_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
    vocab_output_path = "/data/shared/github/lutesAI2025/naufal/go_vocab_full.json"
    model_output_path = "/data/shared/github/lutesAI2025/naufal/bigbird_finetuned_full.pt"

    print("[INFO] Parsing GO DAG...")
    go_graph = extract_go_graph(obo_path)

    print("[INFO] Loading dataset...")
    dataset = ProteinFunctionDataset(embedding_dir, go_mapping_file, go_graph)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = BigBirdProteinModel(input_dim=1541, target_dim=dataset.num_labels, max_len=1913).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=min_lr)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        epoch_loss = 0.0
        for i, (x, attn_mask, y) in enumerate(tqdm(dataloader)):
            x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # AMP context
                preds = model(x, attn_mask)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i == 0 or (i + 1) % 500 == 0:
                print(f"[✓] Trained {i + 1:,} proteins")

        avg_loss = epoch_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), model_output_path)
    print(f"[✓] Model saved to {model_output_path}")

    with open(vocab_output_path, "w") as f:
        json.dump(dataset.go_vocab, f)
    print(f"[✓] Saved GO vocabulary to {vocab_output_path}")

