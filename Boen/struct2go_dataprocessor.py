#!/usr/bin/env python3
"""
Data processor for Struct2GO
Converts FASTA sequences and PDB structures into the required pickle format
Processes GO annotations from OBO and GAF files
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
import requests
from collections import defaultdict
warnings.filterwarnings('ignore')

class Struct2GODataProcessor:
    def __init__(self, fasta_dir, pdb_dir, output_dir, obo_file=None, gaf_file=None, cmap_thresh=10.0):
        """
        Initialize the data processor
        
        Args:
            fasta_dir: Directory containing FASTA files
            pdb_dir: Directory containing PDB.gz files  
            output_dir: Output directory for processed data
            obo_file: Path to GO OBO file (optional)
            gaf_file: Path to GO GAF annotation file (optional)
            cmap_thresh: Distance threshold for contact map (default: 10.0 Å)
        """
        self.fasta_dir = Path(fasta_dir)
        self.pdb_dir = Path(pdb_dir)
        self.output_dir = Path(output_dir)
        self.obo_file = Path(obo_file) if obo_file else None
        self.gaf_file = Path(gaf_file) if gaf_file else None
        self.cmap_thresh = cmap_thresh
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GO term processing
        self.go_terms = {}
        self.protein_go_annotations = defaultdict(set)
        
        # Amino acid vocabulary for one-hot encoding
        self.aa_chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
                        'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
        self.vocab_size = len(self.aa_chars)
        self.vocab_embed = dict(zip(self.aa_chars, range(self.vocab_size)))
        
        # Create one-hot encoding matrix
        self.vocab_one_hot = np.zeros((self.vocab_size, self.vocab_size), int)
        for _, val in self.vocab_embed.items():
            self.vocab_one_hot[val, val] = 1

    def download_goa_file(self, organism="uniprot"):
        """Download GOA file if not provided"""
        base_url = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/"
        
        if organism == "human":
            filename = "gene_association.goa_human.gz"
        elif organism == "uniprot":
            filename = "gene_association.goa_uniprot.gz"
        else:
            filename = f"gene_association.goa_{organism}.gz"
            
        url = base_url + filename
        local_path = self.output_dir / filename
        
        if not local_path.exists():
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url.replace('ftp://', 'http://'))
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return None
        
        return local_path

    def parse_obo_file(self):
        """Parse GO OBO file to extract term information"""
        if not self.obo_file or not self.obo_file.exists():
            print("No OBO file provided, skipping GO term parsing")
            return
            
        print("Parsing GO OBO file...")
        current_term = None
        
        with open(self.obo_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line == "[Term]":
                    current_term = {}
                elif line.startswith("id: "):
                    if current_term is not None:
                        current_term['id'] = line[4:]
                elif line.startswith("name: "):
                    if current_term is not None:
                        current_term['name'] = line[6:]
                elif line.startswith("namespace: "):
                    if current_term is not None:
                        current_term['namespace'] = line[11:]
                elif line == "" and current_term is not None:
                    # End of term
                    if 'id' in current_term:
                        self.go_terms[current_term['id']] = current_term
                    current_term = None
        
        print(f"Parsed {len(self.go_terms)} GO terms")

    def parse_gaf_file(self):
        """Parse GAF file to extract protein-GO associations"""
        if not self.gaf_file or not self.gaf_file.exists():
            print("No GAF file provided, downloading default...")
            self.gaf_file = self.download_goa_file("uniprot")
            
        if not self.gaf_file:
            print("Could not obtain GAF file, skipping GO annotations")
            return
            
        print("Parsing GAF file...")
        
        if self.gaf_file.suffix == '.gz':
            opener = gzip.open
        else:
            opener = open
            
        protein_annotations = defaultdict(set)
        
        with opener(self.gaf_file, 'rt') as f:
            for line in f:
                if line.startswith('!'):  # Skip comment lines
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 17:
                    continue
                    
                db = fields[0]
                protein_id = fields[1]
                go_term = fields[4]
                evidence_code = fields[6]
                
                # Only include certain evidence codes (exclude IEA for higher quality)
                # You can modify this based on your needs
                if evidence_code not in ['IEA']:  # Exclude electronic annotations
                    protein_annotations[protein_id].add(go_term)
        
        self.protein_go_annotations = protein_annotations
        print(f"Parsed annotations for {len(protein_annotations)} proteins")

    def create_go_label_matrix(self, protein_list, go_terms_subset=None, namespace_filter=None):
        """Create binary label matrix for GO terms"""
        if not self.go_terms or not self.protein_go_annotations:
            print("No GO data available, creating dummy labels")
            return {protein_id: torch.zeros(273) for protein_id in protein_list}
        
        # Filter GO terms if needed
        if namespace_filter:
            filtered_terms = {
                term_id: term_data for term_id, term_data in self.go_terms.items()
                if term_data.get('namespace') == namespace_filter
            }
        else:
            filtered_terms = self.go_terms
            
        if go_terms_subset:
            filtered_terms = {
                term_id: term_data for term_id, term_data in filtered_terms.items()
                if term_id in go_terms_subset
            }
        
        # Create term index mapping
        term_list = sorted(filtered_terms.keys())
        term_to_idx = {term: idx for idx, term in enumerate(term_list)}
        
        # Create label matrix
        protein_labels = {}
        for protein_id in protein_list:
            labels = torch.zeros(len(term_list))
            
            # Get annotations for this protein
            protein_terms = self.protein_go_annotations.get(protein_id, set())
            
            for term in protein_terms:
                if term in term_to_idx:
                    labels[term_to_idx[term]] = 1
                    
            protein_labels[protein_id] = labels
        
        print(f"Created label matrix with {len(term_list)} GO terms")
        
        # Save term mapping
        with open(self.output_dir / 'go_term_mapping.pkl', 'wb') as f:
            pickle.dump({
                'term_to_idx': term_to_idx,
                'idx_to_term': {idx: term for term, idx in term_to_idx.items()},
                'term_list': term_list,
                'namespace_filter': namespace_filter
            }, f)
        
        return protein_labels

    def seq2onehot(self, seq):
        """Convert amino acid sequence to one-hot encoding"""
        embed_x = [self.vocab_embed.get(aa, self.vocab_embed['X']) for aa in seq]
        seqs_x = np.array([self.vocab_one_hot[j, :] for j in embed_x])
        return seqs_x

    def load_fasta_sequences(self):
        """Load all FASTA sequences from directory"""
        sequences = {}
        
        # Process individual FASTA files
        fasta_files = list(self.fasta_dir.glob("*.fasta")) + list(self.fasta_dir.glob("*.fa"))
        
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

    def load_pdb_structure(self, pdb_file):
        """Load PDB structure and extract distance matrix and sequence"""
        try:
            # Handle compressed files
            if pdb_file.suffix == '.gz':
                with gzip.open(pdb_file, 'rt') as f:
                    pdb_content = f.read()
                # Write to temporary file for PDB parser
                temp_pdb = pdb_file.parent / f"temp_{pdb_file.stem}"
                with open(temp_pdb, 'w') as f:
                    f.write(pdb_content)
                pdb_path = temp_pdb
            else:
                pdb_path = pdb_file
            
            # Parse PDB structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_file.stem, pdb_path)
            
            # Extract CA atoms and sequence
            residues = []
            sequence = ""
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.has_id('CA'):  # Only consider residues with CA atoms
                            residues.append(residue)
                            # Convert 3-letter to 1-letter amino acid code
                            aa_3to1 = {
                                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                                'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                                'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                                'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                            }
                            sequence += aa_3to1.get(residue.get_resname(), 'X')
            
            # Calculate distance matrix
            n_residues = len(residues)
            distances = np.zeros((n_residues, n_residues))
            
            for i in range(n_residues):
                for j in range(n_residues):
                    if i != j:
                        ca1 = residues[i]['CA'].get_coord()
                        ca2 = residues[j]['CA'].get_coord()
                        distances[i, j] = np.linalg.norm(ca1 - ca2)
            
            # Clean up temporary file
            if pdb_file.suffix == '.gz' and temp_pdb.exists():
                temp_pdb.unlink()
                
            return distances, sequence
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            return None, None

    def create_contact_map_and_graph(self, distances):
        """Create contact map and DGL graph from distance matrix"""
        # Create adjacency matrix (contact map)
        contact_map = (distances < self.cmap_thresh) & (distances > 0)
        
        # Create edge list for graph
        edges = []
        for i in range(len(contact_map)):
            for j in range(len(contact_map)):
                if contact_map[i, j]:
                    edges.append([i, j])
        
        edges = np.array(edges)
        
        # Create DGL graph
        if len(edges) > 0:
            src, dst = edges[:, 0], edges[:, 1]
            graph = dgl.graph((src, dst))
            graph = dgl.add_self_loop(graph)  # Add self-loops
        else:
            # Handle proteins with no contacts (create self-loop only graph)
            n_nodes = len(contact_map)
            graph = dgl.graph((list(range(n_nodes)), list(range(n_nodes))))
        
        return contact_map, graph

    def process_all_data(self):
        """Process all FASTA and PDB data"""
        # Parse GO data if available
        if self.obo_file:
            self.parse_obo_file()
        if self.gaf_file or True:  # Always try to get GAF data
            self.parse_gaf_file()
        
        # Load sequences
        sequences = self.load_fasta_sequences()
        
        # Initialize data containers
        protein_graphs = {}
        protein_sequences = {}
        protein_node_features = {}
        protein_contact_maps = {}
        
        # Process PDB files
        pdb_files = list(self.pdb_dir.glob("*.pdb")) + list(self.pdb_dir.glob("*.pdb.gz"))
        
        processed_count = 0
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            # Extract protein ID from filename
            protein_id = pdb_file.stem.replace('.pdb', '')
            
            # Load structure
            distances, pdb_sequence = self.load_pdb_structure(pdb_file)
            
            if distances is None:
                continue
                
            # Use FASTA sequence if available, otherwise use PDB sequence
            if protein_id in sequences:
                sequence = sequences[protein_id]
            else:
                sequence = pdb_sequence
                
            if not sequence:
                continue
                
            # Ensure sequence and structure match in length
            if len(sequence) != len(distances):
                print(f"Length mismatch for {protein_id}: seq={len(sequence)}, struct={len(distances)}")
                # Use the shorter length
                min_len = min(len(sequence), len(distances))
                sequence = sequence[:min_len]
                distances = distances[:min_len, :min_len]
            
            # Create contact map and graph
            contact_map, graph = self.create_contact_map_and_graph(distances)
            
            # Create one-hot encoded sequence features
            sequence_onehot = self.seq2onehot(sequence)
            
            # Store processed data
            protein_graphs[protein_id] = graph
            protein_sequences[protein_id] = sequence
            protein_node_features[protein_id] = torch.FloatTensor(sequence_onehot)
            protein_contact_maps[protein_id] = contact_map
            
            processed_count += 1
            
        print(f"Successfully processed {processed_count} proteins")
        
        return protein_graphs, protein_sequences, protein_node_features, protein_contact_maps

    def save_processed_data(self, protein_graphs, protein_sequences, protein_node_features, protein_contact_maps):
        """Save processed data to pickle files"""
        
        # Save individual components
        with open(self.output_dir / 'protein_graphs.pkl', 'wb') as f:
            pickle.dump(protein_graphs, f)
            
        with open(self.output_dir / 'protein_sequences.pkl', 'wb') as f:
            pickle.dump(protein_sequences, f)
            
        with open(self.output_dir / 'protein_node_features.pkl', 'wb') as f:
            pickle.dump(protein_node_features, f)
            
        with open(self.output_dir / 'protein_contact_maps.pkl', 'wb') as f:
            pickle.dump(protein_contact_maps, f)
        
        # Create GO labels
        protein_list = list(protein_graphs.keys())
        
        # You can specify namespace filter for specific GO aspects:
        # 'molecular_function' - for MF (typically ~273 terms in Struct2GO papers)
        # 'biological_process' - for BP (typically ~1000+ terms)
        # 'cellular_component' - for CC (typically ~200+ terms)
        # None - for all aspects
        
        mf_labels = self.create_go_label_matrix(
            protein_list, 
            namespace_filter='molecular_function'
        )
        bp_labels = self.create_go_label_matrix(
            protein_list, 
            namespace_filter='biological_process'
        )
        cc_labels = self.create_go_label_matrix(
            protein_list, 
            namespace_filter='cellular_component'
        )
        
        # Save labels by aspect
        with open(self.output_dir / 'protein_labels_mf.pkl', 'wb') as f:
            pickle.dump(mf_labels, f)
        with open(self.output_dir / 'protein_labels_bp.pkl', 'wb') as f:
            pickle.dump(bp_labels, f)
        with open(self.output_dir / 'protein_labels_cc.pkl', 'wb') as f:
            pickle.dump(cc_labels, f)
        
        print(f"Saved processed data to {self.output_dir}")
        print("Files created:")
        print("- protein_graphs.pkl: DGL graphs for each protein")
        print("- protein_sequences.pkl: Amino acid sequences")
        print("- protein_node_features.pkl: One-hot encoded sequence features")
        print("- protein_contact_maps.pkl: Contact maps")
        print("- protein_labels_mf.pkl: Molecular function GO labels")
        print("- protein_labels_bp.pkl: Biological process GO labels") 
        print("- protein_labels_cc.pkl: Cellular component GO labels")
        print("- go_term_mapping.pkl: GO term index mappings")

    def create_dataset_file(self, protein_graphs, protein_node_features, protein_labels, branch='mf'):
        """Create dataset file compatible with Struct2GO MyDataSet class"""
        
        # Create the data structure expected by MyDataSet
        emb_graph = protein_graphs
        emb_seq_feature = protein_node_features  
        emb_label = protein_labels
        
        # Save as the format expected by Struct2GO
        with open(self.output_dir / f'emb_graph_{branch}.pkl', 'wb') as f:
            pickle.dump(emb_graph, f)
            
        with open(self.output_dir / f'emb_seq_feature_{branch}.pkl', 'wb') as f:
            pickle.dump(emb_seq_feature, f)
            
        with open(self.output_dir / f'emb_label_{branch}.pkl', 'wb') as f:
            pickle.dump(emb_label, f)
        
        print(f"Created Struct2GO compatible dataset files for {branch.upper()}:")
        print(f"- emb_graph_{branch}.pkl")
        print(f"- emb_seq_feature_{branch}.pkl") 
        print(f"- emb_label_{branch}.pkl")

def main():
    parser = argparse.ArgumentParser(description='Process protein data for Struct2GO')
    parser.add_argument('--fasta_dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data')
    parser.add_argument('--obo_file', help='Path to GO OBO file (e.g., GO_June_1_2025.obo)')
    parser.add_argument('--gaf_file', help='Path to GO GAF annotation file')
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help='Contact map distance threshold (Å)')
    parser.add_argument('--download_gaf', action='store_true', help='Download GAF file automatically')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = Struct2GODataProcessor(
        fasta_dir=args.fasta_dir,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        obo_file=args.obo_file,
        gaf_file=args.gaf_file,
        cmap_thresh=args.cmap_thresh
    )
    
    # Process all data
    protein_graphs, protein_sequences, protein_node_features, protein_contact_maps = processor.process_all_data()
    
    if not protein_graphs:
        print("No proteins were successfully processed!")
        return
    
    # Save processed data
    processor.save_processed_data(protein_graphs, protein_sequences, protein_node_features, protein_contact_maps)
    
    # Create dataset files compatible with Struct2GO for each GO aspect
    # Load the saved labels
    with open(processor.output_dir / 'protein_labels_mf.pkl', 'rb') as f:
        mf_labels = pickle.load(f)
    with open(processor.output_dir / 'protein_labels_bp.pkl', 'rb') as f:
        bp_labels = pickle.load(f)
    with open(processor.output_dir / 'protein_labels_cc.pkl', 'rb') as f:
        cc_labels = pickle.load(f)
    
    # Create Struct2GO format files for each aspect
    processor.create_dataset_file(protein_graphs, protein_node_features, mf_labels, 'mf')
    processor.create_dataset_file(protein_graphs, protein_node_features, bp_labels, 'bp') 
    processor.create_dataset_file(protein_graphs, protein_node_features, cc_labels, 'cc')
    
    print(f"\nProcessing complete! Processed {len(protein_graphs)} proteins.")
    print(f"Data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Use the emb_graph_mf.pkl, emb_seq_feature_mf.pkl, emb_label_mf.pkl files for molecular function prediction")
    print("2. Use the emb_graph_bp.pkl, emb_seq_feature_bp.pkl, emb_label_bp.pkl files for biological process prediction")
    print("3. Use the emb_graph_cc.pkl, emb_seq_feature_cc.pkl, emb_label_cc.pkl files for cellular component prediction")

if __name__ == "__main__":
    main()