import torch
import sys
import os

# Read first protein ID
with open('my_sequences_processed.txt', 'r') as f:
    first_protein = f.readline().strip()

print(f"First protein ID: '{first_protein}'")

filepath = f'data/{first_protein}.pt'
print(f"Looking for file: {filepath}")

if not os.path.exists(filepath):
    print(f"ERROR: File {filepath} does not exist!")
    # Let's see what files are actually in data/
    print("Files in data/:")
    data_files = [f for f in os.listdir('data/') if f.endswith('.pt')]
    print(data_files[:5])  # Show first 5
    sys.exit(1)

try:
    print("Loading data...")
    data = torch.load(filepath)
    print('Keys:', list(data.keys()))
    print('x shape:', data['x'].shape)
    print('seq shape:', data['seq'].shape)
    print('edge_index shape:', data['edge_index'].shape)
    print('edge_index min/max:', data['edge_index'].min().item(), data['edge_index'].max().item())
    print('pssm shape:', data['pssm'].shape)
    print('seq_embed shape:', data['seq_embed'].shape)
    
    # Check if edge indices are valid
    seq_len = data['x'].shape[0]
    edge_max = data['edge_index'].max().item()
    
    if edge_max >= seq_len:
        print(f"ERROR: edge_index max ({edge_max}) >= sequence length ({seq_len})")
    else:
        print("Data looks valid!")
        
except Exception as e:
    print(f"ERROR loading {filepath}: {e}")
    import traceback
    traceback.print_exc()