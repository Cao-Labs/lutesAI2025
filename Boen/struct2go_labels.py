#!/usr/bin/env python3
"""
Add GO label networks to existing processed data
This script loads your already processed PDB data and adds the missing GO components
"""

import pickle
import pandas as pd
import numpy as np
import torch
import dgl
from pathlib import Path
import argparse
from collections import defaultdict

def load_go_obo(obo_file):
    """Parse GO OBO file to get term relationships"""
    print(f"Loading GO relationships from {obo_file}")
    
    go_terms = {}
    is_a = defaultdict(list)
    part_of = defaultdict(list)
    namespace = defaultdict(str)
    
    current_term = None
    
    with open(obo_file, 'r') as f:
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
            elif line.startswith("is_a: "):
                if current_term is not None:
                    parent = line[6:].split()[0]  # Get GO:XXXXXXX part
                    is_a[current_term.get('id', '')].append(parent)
            elif line.startswith("relationship: part_of "):
                if current_term is not None:
                    parent = line[22:].split()[0]  # Get GO:XXXXXXX part
                    part_of[current_term.get('id', '')].append(parent)
            elif line == "" and current_term is not None:
                # End of term
                if 'id' in current_term:
                    go_terms[current_term['id']] = current_term
                    namespace[current_term['id']] = current_term.get('namespace', '')
                current_term = None
    
    # Combine is_a and part_of relationships
    for term in part_of:
        is_a[term].extend(part_of[term])
    
    print(f"Loaded {len(go_terms)} GO terms")
    return go_terms, is_a, namespace

def load_go_mappings_from_csv(source_data_dir):
    """Load GO term mappings from CSV files"""
    print(f"Loading GO mappings from {source_data_dir}")
    
    mappings = {}
    
    for ontology in ['bp', 'mf', 'cc']:
        go_file = source_data_dir / f'gos_{ontology}.csv'
        
        if not go_file.exists():
            print(f"Warning: {go_file} not found")
            continue
            
        try:
            df = pd.read_csv(go_file)
            
            # Get GO terms column
            go_column = f'{ontology.upper()}-GO'
            if go_column not in df.columns:
                possible_cols = [col for col in df.columns if 'GO' in col.upper()]
                if possible_cols:
                    go_column = possible_cols[0]
                else:
                    print(f"Warning: No GO column found in {go_file}")
                    continue
            
            # Get unique terms and create mappings
            unique_terms = sorted(df[go_column].unique())
            term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
            
            mappings[ontology.upper()] = {
                'terms': unique_terms,
                'term_to_idx': term_to_idx,
                'idx_to_term': {idx: term for term, idx in term_to_idx.items()}
            }
            
            print(f"  {ontology.upper()}: {len(unique_terms)} terms")
            
        except Exception as e:
            print(f"Error loading {go_file}: {e}")
            continue
    
    return mappings

def create_label_network(go_terms, is_a_relations, ontology_terms):
    """Create label network (GO hierarchy) for specific ontology"""
    print(f"Creating label network for {len(ontology_terms)} terms...")
    
    # Create term to index mapping
    term_to_idx = {term: idx for idx, term in enumerate(ontology_terms)}
    
    # Create DGL graph
    num_nodes = len(ontology_terms)
    
    # Find edges based on is_a relationships
    edges_src = []
    edges_dst = []
    
    for child_term in ontology_terms:
        if child_term in is_a_relations:
            child_idx = term_to_idx[child_term]
            
            for parent_term in is_a_relations[child_term]:
                if parent_term in term_to_idx:
                    parent_idx = term_to_idx[parent_term]
                    edges_src.append(child_idx)
                    edges_dst.append(parent_idx)
    
    # Create graph
    if edges_src:
        graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
    else:
        # If no relationships found, create empty graph
        graph = dgl.graph(([], []), num_nodes=num_nodes)
    
    # Add self loops
    graph = dgl.add_self_loop(graph)
    
    print(f"  Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    return graph

def create_dummy_labels(protein_ids, num_labels):
    """Create dummy labels for inference (all zeros)"""
    dummy_labels = {}
    
    for protein_id in protein_ids:
        # Create zero labels for inference
        labels = torch.zeros(num_labels, dtype=torch.float32)
        dummy_labels[protein_id] = labels
    
    return dummy_labels

def main():
    parser = argparse.ArgumentParser(description='Add GO label networks to existing processed data')
    parser.add_argument('--processed_data_dir', required=True, help='Directory with existing processed data')
    parser.add_argument('--source_data_dir', required=True, help='Directory with GO source data (gos_*.csv)')
    parser.add_argument('--go_obo_file', help='Path to go.obo file (optional)')
    parser.add_argument('--output_dir', help='Output directory (default: same as processed_data_dir)')
    
    args = parser.parse_args()
    
    processed_data_dir = Path(args.processed_data_dir)
    source_data_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else processed_data_dir
    
    print("Loading existing processed data...")
    
    # Load existing data
    try:
        with open(processed_data_dir / 'emb_graph_test.pkl', 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} protein graphs")
    except FileNotFoundError:
        print("Error: emb_graph_test.pkl not found. Make sure you're pointing to the right directory.")
        return
    
    # Get protein IDs
    protein_ids = list(graphs.keys())
    print(f"Protein IDs: {protein_ids[:5]}...")
    
    # Load GO mappings from CSV files
    go_mappings = load_go_mappings_from_csv(source_data_dir)
    
    if not go_mappings:
        print("Error: No GO mappings loaded. Check your source data directory.")
        return
    
    # Load GO OBO file if provided
    go_terms = {}
    is_a_relations = {}
    
    if args.go_obo_file and Path(args.go_obo_file).exists():
        go_terms, is_a_relations, namespace = load_go_obo(args.go_obo_file)
    else:
        print("No GO OBO file provided, creating simple label networks...")
    
    # Create label networks and dummy labels for each ontology
    for ontology in ['BP', 'MF', 'CC']:
        if ontology not in go_mappings:
            print(f"Skipping {ontology} - no data found")
            continue
            
        print(f"\nProcessing {ontology} ontology...")
        
        ontology_terms = go_mappings[ontology]['terms']
        num_terms = len(ontology_terms)
        
        # Create label network
        if go_terms and is_a_relations:
            # Use GO hierarchy
            label_network = create_label_network(go_terms, is_a_relations, ontology_terms)
        else:
            # Create simple identity network
            label_network = torch.eye(num_terms, dtype=torch.float32)
            print(f"  Created identity matrix ({num_terms}x{num_terms})")
        
        # Create dummy labels for inference
        dummy_labels = create_dummy_labels(protein_ids, num_terms)
        
        # Save files in the format expected by training code
        ontology_lower = ontology.lower()
        
        # Save label network
        label_network_file = output_dir / f'label_{ontology_lower}_network.pkl'
        with open(label_network_file, 'wb') as f:
            pickle.dump(label_network, f)
        print(f"  Saved: {label_network_file}")
        
        # Save dummy labels
        labels_file = output_dir / f'emb_label_{ontology_lower}.pkl'
        with open(labels_file, 'wb') as f:
            pickle.dump(dummy_labels, f)
        print(f"  Saved: {labels_file}")
        
        # Save GO term mapping
        mapping_file = output_dir / f'go_term_mapping_{ontology_lower}.pkl'
        with open(mapping_file, 'wb') as f:
            pickle.dump(go_mappings[ontology], f)
        print(f"  Saved: {mapping_file}")
    
    print(f"\nCompleted! Added GO label networks to {output_dir}")
    print("\nFiles created:")
    print("- label_bp_network.pkl, label_mf_network.pkl, label_cc_network.pkl")
    print("- emb_label_bp.pkl, emb_label_mf.pkl, emb_label_cc.pkl")
    print("- go_term_mapping_bp.pkl, go_term_mapping_mf.pkl, go_term_mapping_cc.pkl")
    print("\nYour existing PDB processing is preserved!")

if __name__ == "__main__":
    main()