import torch
import sys

def validate_pt_file(filepath):
    try:
        data = torch.load(filepath)
        seq_len = data['x'].shape[0]
        edge_max = data['edge_index'].max().item()
        
        if edge_max >= seq_len:
            print(f"ERROR in {filepath}: edge_index max ({edge_max}) >= sequence length ({seq_len})")
            return False
        return True
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return False

# Test first file
first_protein = open('my_sequences_processed.txt').readline().strip()
validate_pt_file(f'data/{first_protein}.pt')