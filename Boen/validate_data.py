import torch
import os

with open('my_sequences_processed.txt', 'r') as f:
    proteins = [line.strip() for line in f.readlines()[:10]]  # First 10

for protein in proteins:
    filepath = f'data/{protein}.pt'
    if not os.path.exists(filepath):
        print(f"MISSING: {filepath}")
        continue
        
    try:
        data = torch.load(filepath)
        seq_len = data['x'].shape[0]
        edge_max = data['edge_index'].max().item() if data['edge_index'].numel() > 0 else -1
        
        if edge_max >= seq_len:
            print(f"INVALID: {protein} - edge_max={edge_max}, seq_len={seq_len}")
        else:
            print(f"VALID: {protein} - edge_max={edge_max}, seq_len={seq_len}")
            
    except Exception as e:
        print(f"ERROR: {protein} - {e}")