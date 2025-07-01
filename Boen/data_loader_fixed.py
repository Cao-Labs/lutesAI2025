import torch
import numpy as np
import os
from torch_geometric.data import Data, Dataset, DataLoader

class contact_data(Data):
    def __cat_dim__(self, key, item):
        if key in ['seq_embed', 'label', 'chain_id']:
            return None
        elif key in ['x', 'seq', 'pssm']:
            return 0  # Concatenate along sequence dimension
        else:
            return super().__cat_dim__(key, item)

class Protein_Gnn_data(Dataset):
    def __init__(self, root, chain_list, transform=None, pre_transform=None):
        super(Protein_Gnn_data, self).__init__(root, transform, pre_transform)
        self.chain_list = open(chain_list).readlines()
        self.chain_ids = [chain.strip() for chain in self.chain_list]
    
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)
    
    def get(self, idx):
        data = torch.load(self.root + '/' + self.chain_ids[idx] + '.pt')
        
        # Validate edge indices
        seq_len = data['x'].shape[0]
        edge_max = data['edge_index'].max().item() if data['edge_index'].numel() > 0 else -1
        
        if edge_max >= seq_len:
            print(f"Warning: Skipping {self.chain_ids[idx]} due to invalid edge indices")
            # Create minimal valid data with self-loops
            edge_index = torch.tensor([[i, i] for i in range(seq_len)], dtype=torch.long).t()
            data['edge_index'] = edge_index
        
        data = contact_data(
            x=data['x'], 
            pssm=data['pssm'], 
            seq=data['seq'], 
            edge_index=data['edge_index'], 
            seq_embed=data['seq_embed'], 
            label=data['label'], 
            chain_id=self.chain_ids[idx]
        )
        return data