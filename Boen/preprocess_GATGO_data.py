#!/usr/bin/env python3
"""
GAT-GO Data Preprocessing Script
Converts FASTA sequences and PDB structures to GAT-GO format
"""

import os
import gzip
import torch
import numpy as np
import tempfile
from Bio import SeqIO
from Bio.PDB import PDBParser
import esm
import argparse
from tqdm import tqdm

# Amino acid to index mapping (20 standard + 5 special tokens)
AA_TO_IDX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, 'U': 21, 'B': 22, 'Z': 23, 'O': 24  # Special tokens
}

def load_esm_model():
    """Load ESM-1b model for generating embeddings"""
    print("Loading ESM-1b model...")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, batch_converter

def parse_fasta(fasta_file):
    """Parse FASTA file and return sequences dictionary"""
    sequences = {}
    print(f"Parsing FASTA file: {fasta_file}")
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    
    print(f"Found {len(sequences)} sequences")
    return sequences

def one_hot_encode_sequence(sequence):
    """Convert amino acid sequence to one-hot encoding"""
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, 25))  # 25 dimensions as expected by GAT-GO
    
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            one_hot[i, AA_TO_IDX[aa]] = 1
        else:
            one_hot[i, AA_TO_IDX['X']] = 1  # Unknown amino acid
    
    return torch.tensor(one_hot, dtype=torch.float32)

def generate_esm_embeddings(sequence, model, batch_converter, device='cpu'):
    """Generate ESM-1b embeddings for a sequence"""
    # Prepare data for ESM
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Extract representations
    token_representations = results["representations"][33]
    sequence_representations = results["representations"][33].mean(1)  # Mean pooling
    
    # Remove special tokens (first and last)
    residue_embeddings = token_representations[0, 1:len(sequence)+1]
    sequence_embedding = sequence_representations[0]
    
    return residue_embeddings.cpu(), sequence_embedding.cpu()

def extract_contact_map(pdb_file, distance_threshold=8.0):
    """Extract contact map from PDB structure"""
    import tempfile
    
    # Handle gzipped PDB files
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as f:
            pdb_content = f.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_f:
            temp_f.write(pdb_content)
            temp_pdb_path = temp_f.name
    else:
        temp_pdb_path = pdb_file
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', temp_pdb_path)
        
        # Get CA atoms
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atoms.append(residue['CA'])
        
        # Calculate distances and create contact map
        n_residues = len(ca_atoms)
        edge_indices = []
        
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                distance = ca_atoms[i] - ca_atoms[j]  # BioPython calculates distance automatically
                if distance <= distance_threshold:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Add both directions
        
        if len(edge_indices) == 0:
            # If no contacts, create self-loops
            edge_indices = [[i, i] for i in range(n_residues)]
        
        return torch.tensor(edge_indices, dtype=torch.long).t()
        
    finally:
        # Clean up temporary file if we created one
        if pdb_file.endswith('.gz') and os.path.exists(temp_pdb_path):
            os.remove(temp_pdb_path)

def generate_dummy_pssm(sequence_length):
    """Generate dummy PSSM (you should replace this with real PSSM generation)"""
    # This creates a dummy PSSM - you might want to use PSI-BLAST for real PSSMs
    dummy_pssm = np.random.randn(sequence_length, 20) * 0.1
    return torch.tensor(dummy_pssm, dtype=torch.float32)

def process_protein(protein_id, sequence, pdb_file, model, batch_converter, output_dir, device='cpu'):
    """Process a single protein and save in GAT-GO format"""
    try:
        print(f"Processing {protein_id}...")
        
        # Generate ESM embeddings
        residue_embeddings, sequence_embedding = generate_esm_embeddings(
            sequence, model, batch_converter, device
        )
        
        # One-hot encode sequence
        one_hot_seq = one_hot_encode_sequence(sequence)
        
        # Extract contact map
        contact_map = extract_contact_map(pdb_file)
        
        # Generate PSSM (dummy for now)
        pssm = generate_dummy_pssm(len(sequence))
        
        # Create dummy label (empty for prediction)
        label = torch.zeros(2752)  # GAT-GO expects 2752 GO terms
        
        # Create data dictionary
        data_dict = {
            'x': residue_embeddings,           # ESM-1b residue embeddings
            'seq': one_hot_seq,                # One-hot encoded sequence
            'pssm': pssm,                      # PSSM matrix
            'edge_index': contact_map,         # Contact map edges
            'seq_embed': sequence_embedding,   # ESM-1b sequence embedding
            'label': label,                    # GO term labels (dummy)
        }
        
        # Save to file
        output_file = os.path.join(output_dir, f"{protein_id}.pt")
        torch.save(data_dict, output_file)
        print(f"Saved {protein_id}.pt")
        
        return True
        
    except Exception as e:
        print(f"Error processing {protein_id}: {str(e)}")
        return False

def create_sequence_list(sequences, output_file):
    """Create sequence ID list file for GAT-GO"""
    with open(output_file, 'w') as f:
        for protein_id in sequences.keys():
            f.write(f"{protein_id}\n")
    print(f"Created sequence list: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for GAT-GO')
    parser.add_argument('--fasta', required=True, help='Path to FASTA file')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', default='data/seq_features', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--seq_list', default='sequence_ids.txt', help='Output sequence list file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ESM model
    device = torch.device(args.device)
    model, batch_converter = load_esm_model()
    model = model.to(device)
    
    # Parse FASTA sequences
    sequences = parse_fasta(args.fasta)
    
    # Process each protein
    successful = 0
    failed = 0
    
    # Get list of all PDB files and create lookup
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb.gz')]
    pdb_lookup = {f.replace('.pdb.gz', ''): f for f in pdb_files}
    print(f"Found {len(pdb_files)} PDB files")
    
    for protein_id, sequence in tqdm(sequences.items()):
        # Find corresponding PDB file
        if protein_id in pdb_lookup:
            pdb_file = os.path.join(args.pdb_dir, pdb_lookup[protein_id])
        else:
            print(f"Warning: No PDB file found for {protein_id}")
            failed += 1
            continue
        
        success = process_protein(
            protein_id, sequence, pdb_file, model, batch_converter, 
            args.output_dir, device
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Create sequence list file
    create_sequence_list(sequences, args.seq_list)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sequence list: {args.seq_list}")

if __name__ == "__main__":
    main()