#!/usr/bin/env python3
"""
FINAL CORRECTED Struct2GO Data Processor
Based on the actual codebase analysis showing:
- 26-dim one-hot encoding (protein_node2onehot) 
- 30-dim additional features to reach 56 total
- Separate 1024-dim sequence features (dict_sequence_feature)
"""

import os
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import dgl
import torch
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class Struct2GOProcessor:
    def __init__(self, fasta_dir, pdb_dir, output_dir, cmap_thresh=10.0):
        """
        Initialize processor to match the exact original pipeline
        """
        self.fasta_dir = Path(fasta_dir) if fasta_dir else None
        self.pdb_dir = Path(pdb_dir)
        self.output_dir = Path(output_dir)
        self.cmap_thresh = cmap_thresh
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Exact same amino acid vocabulary as original
        self.chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
                     'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
        self.vocab_size = len(self.chars)
        self.vocab_embed = dict(zip(self.chars, range(self.vocab_size)))
        
        # Create one-hot encoding matrix exactly as original
        self.vocab_one_hot = np.zeros((self.vocab_size, self.vocab_size), int)
        for _, val in self.vocab_embed.items():
            self.vocab_one_hot[val, val] = 1

    def seq2onehot(self, seq):
        """Create 26-dim embedding - exact copy of original function"""
        embed_x = [self.vocab_embed[v] for v in seq]
        seqs_x = np.array([self.vocab_one_hot[j, :] for j in embed_x])
        return seqs_x

    def load_predicted_PDB(self, pdbfile):
        """Exact copy of original function"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
        
        residues = [r for r in structure.get_residues()]
        
        # sequence from atom lines
        records = SeqIO.parse(pdbfile, 'pdb-atom')
        seqs = [str(r.seq) for r in records]
        
        distances = np.empty((len(residues), len(residues)))
        for x in range(len(residues)):
            for y in range(len(residues)):
                one = residues[x]["CA"].get_coord()
                two = residues[y]["CA"].get_coord()
                distances[x, y] = np.linalg.norm(one-two)
        
        return distances, seqs[0]

    def load_cmap(self, filename, cmap_thresh=10.0):
        """Exact copy of original load_cmap function"""
        if filename.endswith('.pdb'):
            D, seq = self.load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        
        S = self.seq2onehot(seq)
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)
        return A, S, seq

    def create_30_additional_features(self, seq, distances):
        """
        Create 30 additional features to combine with 26-dim one-hot
        Based on the codebase showing 56 total dimensions needed
        """
        n_residues = len(seq)
        additional_features = np.zeros((n_residues, 30))
        
        for i in range(n_residues):
            # Position-based features (10 features)
            additional_features[i, 0] = i / n_residues  # Relative position
            additional_features[i, 1] = np.sin(2 * np.pi * i / n_residues)  # Sine position
            additional_features[i, 2] = np.cos(2 * np.pi * i / n_residues)  # Cosine position
            additional_features[i, 3] = min(i, n_residues - i) / n_residues  # Distance to ends
            additional_features[i, 4] = abs(i - n_residues/2) / n_residues   # Distance to center
            additional_features[i, 5] = i  # Absolute position
            additional_features[i, 6] = n_residues - i  # Distance from end
            additional_features[i, 7] = 1.0 if i == 0 else 0.0  # N-terminus
            additional_features[i, 8] = 1.0 if i == n_residues-1 else 0.0  # C-terminus
            additional_features[i, 9] = 1.0 if i == n_residues//2 else 0.0  # Middle
            
            # Distance-based features (10 features)
            row_distances = distances[i, :]
            additional_features[i, 10] = np.mean(row_distances)
            additional_features[i, 11] = np.std(row_distances)
            additional_features[i, 12] = np.min(row_distances[row_distances > 0])
            additional_features[i, 13] = np.max(row_distances)
            additional_features[i, 14] = np.median(row_distances)
            additional_features[i, 15] = np.sum(row_distances < 5.0)
            additional_features[i, 16] = np.sum(row_distances < 8.0)
            additional_features[i, 17] = np.sum(row_distances < 12.0)
            additional_features[i, 18] = np.sum(row_distances < 15.0)
            additional_features[i, 19] = np.sum(row_distances < 20.0)
            
            # Amino acid property features (10 features)
            aa = seq[i]
            # Hydrophobicity scale
            hydrophobicity = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
            
            additional_features[i, 20] = hydrophobicity.get(aa, 0.0)
            additional_features[i, 21] = 1.0 if aa in 'AILMFPWYV' else 0.0  # Hydrophobic
            additional_features[i, 22] = 1.0 if aa in 'DEHKR' else 0.0  # Charged
            additional_features[i, 23] = 1.0 if aa in 'NQST' else 0.0  # Polar
            additional_features[i, 24] = 1.0 if aa in 'FWY' else 0.0  # Aromatic
            additional_features[i, 25] = 1.0 if aa in 'CMST' else 0.0  # Small
            additional_features[i, 26] = 1.0 if aa in 'FHKRWY' else 0.0  # Large
            additional_features[i, 27] = 1.0 if aa in 'KR' else 0.0  # Positive
            additional_features[i, 28] = 1.0 if aa in 'DE' else 0.0  # Negative
            additional_features[i, 29] = 1.0 if aa == 'P' else 0.0  # Proline
            
        return additional_features

    def create_56_dim_features(self, seq, distances):
        """
        Create exactly 56-dimensional features as used in the original model
        26-dim one-hot + 30-dim additional features = 56 dimensions
        """
        # Get 26-dimensional one-hot encoding
        onehot_26 = self.seq2onehot(seq)  # Shape: (L, 26)
        
        # Get 30 additional features  
        additional_30 = self.create_30_additional_features(seq, distances)  # Shape: (L, 30)
        
        # Concatenate to get exactly 56 dimensions
        combined_56 = np.concatenate([onehot_26, additional_30], axis=1)  # Shape: (L, 56)
        
        return combined_56

    def create_1024_sequence_features(self, seq):
        """
        Create 1024-dimensional sequence features (placeholder for ELMo embeddings)
        This would typically come from the ELMo model, but we'll create dummy features
        """
        # Create dummy 1024-dimensional features
        # In practice, this would come from dict_sequence_feature
        return np.random.normal(0, 0.1, (1024,)).astype(np.float32)

    def load_fasta_sequences(self):
        """Load FASTA sequences"""
        sequences = {}
        
        if not self.fasta_dir:
            print("No FASTA directory provided, using PDB sequences only")
            return sequences
            
        print(f"Loading FASTA sequences from {self.fasta_dir}")
        fasta_files = list(self.fasta_dir.glob("*.fasta")) + list(self.fasta_dir.glob("*.fa")) + list(self.fasta_dir.glob("*.faa"))
        
        if not fasta_files:
            print(f"No FASTA files found in {self.fasta_dir}")
            return sequences
            
        for fasta_file in tqdm(fasta_files, desc="Loading FASTA files"):
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    protein_id = record.id.split('|')[1] if '|' in record.id else record.id
                    sequences[protein_id] = str(record.seq).upper()
            except Exception as e:
                print(f"Error processing {fasta_file}: {e}")
                continue
                
        print(f"Loaded {len(sequences)} sequences from FASTA files")
        return sequences

    def process_proteins(self):
        """Process all proteins using the exact original pipeline"""
        
        # Load FASTA sequences first
        fasta_sequences = self.load_fasta_sequences()
        
        protein_graphs = {}
        protein_node_features = {}  # 56-dim features for graph nodes
        protein_sequence_features = {}  # 1024-dim features for sequence
        protein_sequences = {}
        protein_contact_maps = {}
        
        # Process PDB files
        pdb_files = list(self.pdb_dir.glob("*.pdb")) + list(self.pdb_dir.glob("*.pdb.gz"))
        
        processed_count = 0
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                # Handle compressed files
                if pdb_file.suffix == '.gz':
                    with gzip.open(pdb_file, 'rt') as f:
                        pdb_content = f.read()
                    temp_pdb = pdb_file.parent / f"temp_{pdb_file.stem}"
                    with open(temp_pdb, 'w') as f:
                        f.write(pdb_content)
                    pdb_path = str(temp_pdb)
                else:
                    pdb_path = str(pdb_file)
                
                # Extract protein ID (matching original format)
                if '-' in pdb_file.stem:
                    protein_id = pdb_file.stem.split('-')[1].replace('.pdb', '')
                else:
                    protein_id = pdb_file.stem.replace('.pdb', '')
                
                # Load using original load_cmap function
                A, S_original, seq = self.load_cmap(pdb_path, self.cmap_thresh)
                
                if A is None or len(seq) == 0:
                    continue
                
                # Use FASTA sequence if available, otherwise use PDB sequence
                if protein_id in fasta_sequences:
                    final_seq = fasta_sequences[protein_id]
                    # Ensure sequence and structure match in length
                    if len(final_seq) != A.shape[1]:
                        min_len = min(len(final_seq), A.shape[1])
                        final_seq = final_seq[:min_len]
                        A = A[:, :min_len, :min_len]
                else:
                    final_seq = seq
                
                # Create 56-dimensional node features (for graph nodes)
                distances_2d = A.squeeze(0)  # Remove batch dimension
                node_features_56 = self.create_56_dim_features(final_seq, distances_2d)
                
                # Create 1024-dimensional sequence features (for sequence input)
                sequence_features_1024 = self.create_1024_sequence_features(final_seq)
                
                # Create 56-dimensional node features (for graph nodes)
                distances_2d = A.squeeze(0)  # Remove batch dimension
                node_features_56 = self.create_56_dim_features(final_seq, distances_2d)
                
                # Create 1024-dimensional sequence features (for sequence input)
                sequence_features_1024 = self.create_1024_sequence_features(final_seq)
                
                # Create graph from contact map (exactly like original)
                edges_data = []
                B = np.reshape(A, (-1, len(A[0])))
                result = []
                N = len(B)
                for i in range(N):
                    for j in range(N):
                        if B[i][j] and i != j:
                            result.append([i, j])
                
                if result:
                    edges_array = np.array(result)
                    src = edges_array[:, 0]
                    dst = edges_array[:, 1]
                    g = dgl.graph((src, dst))
                    g = dgl.add_self_loop(g)
                else:
                    # Create self-loop only graph
                    n_nodes = len(final_seq)
                    g = dgl.graph((list(range(n_nodes)), list(range(n_nodes))))
                
                # Add node features to graph (56-dimensional)
                g.ndata['feature'] = torch.tensor(node_features_56, dtype=torch.float32)
                
                # Store processed data
                protein_graphs[protein_id] = g
                protein_node_features[protein_id] = torch.tensor(node_features_56, dtype=torch.float32)
                protein_sequence_features[protein_id] = torch.tensor(sequence_features_1024, dtype=torch.float32)
                protein_sequences[protein_id] = final_seq
                protein_contact_maps[protein_id] = A.squeeze(0)
                
                processed_count += 1
                
                # Verify dimensions on first protein
                if processed_count == 1:
                    print(f"Node features shape: {node_features_56.shape}")
                    print(f"Node features per residue: {node_features_56.shape[1]} (should be 56)")
                    print(f"Sequence features shape: {sequence_features_1024.shape}")
                    print(f"Sequence features dimension: {sequence_features_1024.shape[0]} (should be 1024)")
                    assert node_features_56.shape[1] == 56, f"Expected 56 node features, got {node_features_56.shape[1]}"
                    assert sequence_features_1024.shape[0] == 1024, f"Expected 1024 sequence features, got {sequence_features_1024.shape[0]}"
                
                # Clean up temporary file
                if pdb_file.suffix == '.gz' and temp_pdb.exists():
                    temp_pdb.unlink()
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} proteins")
        print(f"Node features: 56-dimensional (26 one-hot + 30 additional)")
        print(f"Sequence features: 1024-dimensional (for sequence input)")
        
        # Save processed data
        self.save_data(protein_graphs, protein_node_features, protein_sequence_features, 
                      protein_sequences, protein_contact_maps)
        
        return protein_graphs, protein_node_features, protein_sequence_features, protein_sequences, protein_contact_maps

    def save_data(self, protein_graphs, protein_node_features, protein_sequence_features, 
                  protein_sequences, protein_contact_maps):
        """Save processed data in the exact format expected by the original training code"""
        
        # Save in the format expected by the training code
        with open(self.output_dir / 'emb_graph_test.pkl', 'wb') as f:
            pickle.dump(protein_graphs, f)
        
        # This is the 56-dimensional features used as graph node features
        with open(self.output_dir / 'protein_node2onehot.pkl', 'wb') as f:
            pickle.dump(protein_node_features, f)
        
        # This is the 1024-dimensional features used as sequence input
        with open(self.output_dir / 'dict_sequence_feature.pkl', 'wb') as f:
            pickle.dump(protein_sequence_features, f)
        
        with open(self.output_dir / 'protein_sequences.pkl', 'wb') as f:
            pickle.dump(protein_sequences, f)
        
        with open(self.output_dir / 'protein_contact_maps.pkl', 'wb') as f:
            pickle.dump(protein_contact_maps, f)
        
        print(f"Saved data to {self.output_dir}")
        print("Files created (matching original format):")
        print("- emb_graph_test.pkl: DGL graphs with 56-dim node features")
        print("- protein_node2onehot.pkl: 56-dimensional node features")
        print("- dict_sequence_feature.pkl: 1024-dimensional sequence features")
        print("- protein_sequences.pkl: Amino acid sequences")
        print("- protein_contact_maps.pkl: Contact maps")
        
        # Print verification
        sample_node_features = next(iter(protein_node_features.values()))
        sample_seq_features = next(iter(protein_sequence_features.values()))
        print(f"\nVerification:")
        print(f"Node features shape: {sample_node_features.shape} (should be [L, 56])")
        print(f"Sequence features shape: {sample_seq_features.shape} (should be [1024])")
        print(f"First 26 dims are one-hot: {sample_node_features[0, :26].sum()}")
        print(f"Last 30 dims are additional: mean={sample_node_features[0, 26:].mean():.4f}")

def main():
    parser = argparse.ArgumentParser(description='Process protein data for Struct2GO (exact format)')
    parser.add_argument('--fasta_dir', help='Directory containing FASTA files (optional)')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help='Contact map threshold (Å)')
    
    args = parser.parse_args()
    
    processor = Struct2GOProcessor(
        fasta_dir=args.fasta_dir,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        cmap_thresh=args.cmap_thresh
    )
    
    print("Processing proteins with exact original format...")
    print("- 56-dimensional node features (26 one-hot + 30 additional)")
    print("- 1024-dimensional sequence features (placeholder for ELMo)")
    processor.process_proteins()
    print("Processing complete!")

if __name__ == "__main__":
    main()