#!/usr/bin/env python3
"""
Modified Data processor for Struct2GO
Loads FASTA sequences and PDB structures first, then optionally processes GO annotations
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

    def process_structures_only(self):
        """Process only FASTA and PDB data without GO annotations"""
        print("Processing structures and sequences only...")
        
        # Load sequences
        sequences = self.load_fasta_sequences()
        
        # Process PDB files
        protein_graphs, protein_sequences, protein_node_features, protein_contact_maps = self.process_pdb_files(sequences)
        
        # Save structural data
        self.save_structural_data(protein_graphs, protein_sequences, protein_node_features, protein_contact_maps)
        
        return protein_graphs, protein_sequences, protein_node_features, protein_contact_maps

    def process_with_annotations(self, protein_graphs=None, protein_sequences=None, protein_node_features=None, protein_contact_maps=None):
        """Process GO annotations and create labeled datasets"""
        print("Processing GO annotations...")
        
        # If no structural data provided, load from files
        if protein_graphs is None:
            print("Loading previously processed structural data...")
            try:
                with open(self.output_dir / 'protein_graphs.pkl', 'rb') as f:
                    protein_graphs = pickle.load(f)
                with open(self.output_dir / 'protein_sequences.pkl', 'rb') as f:
                    protein_sequences = pickle.load(f)
                with open(self.output_dir / 'protein_node_features.pkl', 'rb') as f:
                    protein_node_features = pickle.load(f)
                with open(self.output_dir / 'protein_contact_maps.pkl', 'rb') as f:
                    protein_contact_maps = pickle.load(f)
            except FileNotFoundError:
                print("No structural data found. Run process_structures_only() first.")
                return None, None, None, None
        
        # Parse GO data with protein filtering for speed
        if self.obo_file:
            self.parse_obo_file()
        if self.gaf_file or True:  # Always try to get GAF data
            # Filter to only proteins we have structures for
            protein_filter = set(protein_graphs.keys())
            # Use organism preference if set
            organism = getattr(self, 'organism', 'human')
            if not self.gaf_file:
                self.gaf_file = self.download_goa_file(organism)
            self.parse_gaf_file(protein_filter=protein_filter)
        
        # Create GO labels and dataset files
        self.create_go_datasets(protein_graphs, protein_node_features)
        
        return protein_graphs, protein_sequences, protein_node_features, protein_contact_maps

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

    def process_pdb_files(self, sequences):
        """Process PDB files and create graphs"""
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

    def save_structural_data(self, protein_graphs, protein_sequences, protein_node_features, protein_contact_maps):
        """Save structural data only"""
        
        # Save individual components
        with open(self.output_dir / 'protein_graphs.pkl', 'wb') as f:
            pickle.dump(protein_graphs, f)
            
        with open(self.output_dir / 'protein_sequences.pkl', 'wb') as f:
            pickle.dump(protein_sequences, f)
            
        with open(self.output_dir / 'protein_node_features.pkl', 'wb') as f:
            pickle.dump(protein_node_features, f)
            
        with open(self.output_dir / 'protein_contact_maps.pkl', 'wb') as f:
            pickle.dump(protein_contact_maps, f)
        
        print(f"Saved structural data to {self.output_dir}")
        print("Files created:")
        print("- protein_graphs.pkl: DGL graphs for each protein")
        print("- protein_sequences.pkl: Amino acid sequences")
        print("- protein_node_features.pkl: One-hot encoded sequence features")
        print("- protein_contact_maps.pkl: Contact maps")

    # ... [Include all the GO processing methods from original code] ...
    
    def download_goa_file(self, organism="human"):  # Changed default to human
        """Download GOA file if not provided"""
        base_url = "http://ftp.ebi.ac.uk/pub/databases/GO/goa/"
        
        if organism == "human":
            filename = "goa_human.gaf.gz"
            url = base_url + "HUMAN/" + filename
        elif organism == "uniprot":
            filename = "goa_uniprot_all.gaf.gz"
            url = base_url + "UNIPROT/" + filename
        else:
            filename = f"goa_{organism}.gaf.gz"
            url = base_url + f"{organism.upper()}/" + filename
            
        local_path = self.output_dir / filename
        
        if not local_path.exists():
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return None
        
        return local_path

    def find_gaf_file(self):
        """Find any GAF file in the output directory"""
        gaf_patterns = [
            "*.gaf.gz", "*.gaf", 
            "gene_association.*.gz", "gene_association.*",
            "goa_*.gz", "goa_*"
        ]
        
        for pattern in gaf_patterns:
            files = list(self.output_dir.glob(pattern))
            if files:
                print(f"Found GAF file: {files[0]}")
                return files[0]
        
        return None

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
            print("No GAF file provided, looking for downloaded files...")
            self.gaf_file = self.find_gaf_file()
            
            if not self.gaf_file:
                print("No GAF file found, downloading default...")
                self.gaf_file = self.download_goa_file("uniprot")
            
        if not self.gaf_file:
            print("Could not obtain GAF file, skipping GO annotations")
            return
            
        print(f"Parsing GAF file: {self.gaf_file}")
        
        # Smart file opening - try gzip first, then regular file
        try:
            with gzip.open(self.gaf_file, 'rt') as test_file:
                test_file.readline()
            opener = gzip.open
            mode = 'rt'
            print("File detected as gzipped")
        except (gzip.BadGzipFile, OSError, UnicodeDecodeError):
            opener = open
            mode = 'r'
            print("File detected as regular text")
            
        protein_annotations = defaultdict(set)
        line_count = 0
        annotation_count = 0
        
        try:
            with opener(self.gaf_file, mode) as f:
                for line in f:
                    line_count += 1
                    
                    if line.startswith('!'):
                        continue
                    
                    if not line.strip():
                        continue
                        
                    fields = line.strip().split('\t')
                    if len(fields) < 15:
                        continue
                        
                    try:
                        db = fields[0]
                        protein_id = fields[1]
                        go_term = fields[4]
                        evidence_code = fields[6]
                        
                        if evidence_code not in ['IEA']:
                            protein_annotations[protein_id].add(go_term)
                            annotation_count += 1
                            
                    except (IndexError, ValueError) as e:
                        continue
                        
                    if line_count % 100000 == 0:
                        print(f"Processed {line_count} lines, found {annotation_count} annotations...")
        
        except Exception as e:
            print(f"Error parsing GAF file: {e}")
            print("Continuing without GO annotations...")
            return
        
        self.protein_go_annotations = protein_annotations
        print(f"Parsed annotations for {len(protein_annotations)} proteins")
        print(f"Total annotations: {annotation_count}")

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
            
            protein_terms = self.protein_go_annotations.get(protein_id, set())
            
            for term in protein_terms:
                if term in term_to_idx:
                    labels[term_to_idx[term]] = 1
                    
            protein_labels[protein_id] = labels
        
        print(f"Created label matrix with {len(term_list)} GO terms")
        
        # Save term mapping
        with open(self.output_dir / f'go_term_mapping_{namespace_filter or "all"}.pkl', 'wb') as f:
            pickle.dump({
                'term_to_idx': term_to_idx,
                'idx_to_term': {idx: term for term, idx in term_to_idx.items()},
                'term_list': term_list,
                'namespace_filter': namespace_filter
            }, f)
        
        return protein_labels

    def create_go_datasets(self, protein_graphs, protein_node_features):
        """Create GO datasets for all aspects"""
        protein_list = list(protein_graphs.keys())
        
        # Create GO labels
        mf_labels = self.create_go_label_matrix(protein_list, namespace_filter='molecular_function')
        bp_labels = self.create_go_label_matrix(protein_list, namespace_filter='biological_process')
        cc_labels = self.create_go_label_matrix(protein_list, namespace_filter='cellular_component')
        
        # Save labels
        with open(self.output_dir / 'protein_labels_mf.pkl', 'wb') as f:
            pickle.dump(mf_labels, f)
        with open(self.output_dir / 'protein_labels_bp.pkl', 'wb') as f:
            pickle.dump(bp_labels, f)
        with open(self.output_dir / 'protein_labels_cc.pkl', 'wb') as f:
            pickle.dump(cc_labels, f)
        
        # Create Struct2GO format files
        self.create_dataset_file(protein_graphs, protein_node_features, mf_labels, 'mf')
        self.create_dataset_file(protein_graphs, protein_node_features, bp_labels, 'bp')
        self.create_dataset_file(protein_graphs, protein_node_features, cc_labels, 'cc')
        
        # Print summary of label counts
        print(f"\nLabel statistics:")
        print(f"MF labels: {len(next(iter(mf_labels.values())))} terms")
        print(f"BP labels: {len(next(iter(bp_labels.values())))} terms") 
        print(f"CC labels: {len(next(iter(cc_labels.values())))} terms")

    def create_label_network(self, protein_labels, branch='mf'):
        """Create label network for hierarchical structure"""
        # Get label dimensions
        sample_labels = next(iter(protein_labels.values()))
        n_labels = len(sample_labels)
        
        # Create identity matrix as basic label network (can be enhanced with GO hierarchy)
        label_network = torch.eye(n_labels, dtype=torch.float32)
        
        # Save label network
        with open(self.output_dir / f'label_{branch}_network.pkl', 'wb') as f:
            pickle.dump(label_network, f)
        
        print(f"Created label network: label_{branch}_network.pkl")
        return label_network

    def create_test_dataset(self, protein_graphs, protein_node_features, protein_labels, branch='mf'):
        """Create test dataset in MyDataSet compatible format"""
        # Prepare data in the format expected by MyDataSet
        # This should match the structure that your training script expects
        
        protein_ids = list(protein_graphs.keys())
        
        # Create dataset structure
        dataset_data = []
        for protein_id in protein_ids:
            graph = protein_graphs[protein_id]
            features = protein_node_features[protein_id]
            labels = protein_labels[protein_id]
            
            # Add protein_id to the data for tracking
            dataset_data.append({
                'graph': graph,
                'features': features, 
                'labels': labels,
                'protein_id': protein_id
            })
        
        # Save as test dataset
        with open(self.output_dir / f'{branch}_test_dataset.pkl', 'wb') as f:
            pickle.dump(dataset_data, f)
        
        print(f"Created test dataset: {branch}_test_dataset.pkl")
        return dataset_data

    def create_dataset_file(self, protein_graphs, protein_node_features, protein_labels, branch='mf'):
        """Create dataset file compatible with Struct2GO MyDataSet class"""
        
        with open(self.output_dir / f'emb_graph_{branch}.pkl', 'wb') as f:
            pickle.dump(protein_graphs, f)
            
        with open(self.output_dir / f'emb_seq_feature_{branch}.pkl', 'wb') as f:
            pickle.dump(protein_node_features, f)
            
        with open(self.output_dir / f'emb_label_{branch}.pkl', 'wb') as f:
            pickle.dump(protein_labels, f)
        
        # Create label network
        label_network = self.create_label_network(protein_labels, branch)
        
        # Create test dataset
        test_dataset = self.create_test_dataset(protein_graphs, protein_node_features, protein_labels, branch)
        
        print(f"Created Struct2GO compatible dataset files for {branch.upper()}:")
        print(f"- emb_graph_{branch}.pkl")
        print(f"- emb_seq_feature_{branch}.pkl") 
        print(f"- emb_label_{branch}.pkl")
        print(f"- label_{branch}_network.pkl")
        print(f"- {branch}_test_dataset.pkl")

def main():
    parser = argparse.ArgumentParser(description='Process protein data for Struct2GO')
    parser.add_argument('--fasta_dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data')
    parser.add_argument('--obo_file', help='Path to GO OBO file')
    parser.add_argument('--gaf_file', help='Path to GO GAF annotation file')
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help='Contact map distance threshold (Å)')
    parser.add_argument('--structures_only', action='store_true', help='Process only structures, skip GO annotations')
    parser.add_argument('--annotations_only', action='store_true', help='Process only GO annotations (requires existing structural data)')
    parser.add_argument('--organism', default='human', choices=['human', 'mouse', 'yeast', 'ecoli', 'uniprot'], 
                       help='Organism for GO annotations (default: human for speed)')
    
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
    
    # Set organism for download
    processor.organism = args.organism
    
    if args.structures_only:
        # Process only structures
        print("Processing structures only...")
        protein_graphs, protein_sequences, protein_node_features, protein_contact_maps = processor.process_structures_only()
        print(f"Structural processing complete! Processed {len(protein_graphs)} proteins.")
        
    elif args.annotations_only:
        # Process only annotations
        print("Processing annotations only...")
        processor.process_with_annotations()
        print("Annotation processing complete!")
        
    else:
        # Process everything
        print("Processing structures first...")
        protein_graphs, protein_sequences, protein_node_features, protein_contact_maps = processor.process_structures_only()
        
        if protein_graphs:
            print("Processing annotations...")
            processor.process_with_annotations(protein_graphs, protein_sequences, protein_node_features, protein_contact_maps)
            print(f"Complete processing finished! Processed {len(protein_graphs)} proteins.")
        else:
            print("No proteins were successfully processed!")

if __name__ == "__main__":
    main()