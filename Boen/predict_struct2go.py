import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import DGL with legacy compatibility
try:
    import dgl
    from dgl.dataloading import GraphDataLoader
    print(f"DGL version: {dgl.__version__}")
except ImportError:
    print("DGL not available")

def predict_ontology(model, graphs, node_features, sequence_features, label_network, device='cpu', ontology_name=''):
    """Predict for a single ontology"""
    
    model.eval()
    all_predictions = []
    protein_ids = list(graphs.keys())
    
    print(f"\nRunning {ontology_name} predictions on {len(protein_ids)} proteins...")
    
    with torch.no_grad():
        for i, protein_id in enumerate(protein_ids):
            try:
                graph = graphs[protein_id].to(device)
                node_feature = node_features[protein_id].to(device)
                sequence_feature = sequence_features[protein_id].to(device)
                
                # Ensure graph has node features
                if 'feature' not in graph.ndata:
                    graph.ndata['feature'] = node_feature
                
                # Create batch of 1
                batched_graph = dgl.batch([graph])
                batched_sequence_feature = sequence_feature.unsqueeze(0)
                
                # Model prediction
                logits = model(batched_graph, batched_sequence_feature, label_network.to(device))
                predictions = torch.sigmoid(logits)
                
                all_predictions.append(predictions.cpu().numpy())
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(protein_ids)} proteins...")
                    
            except Exception as e:
                print(f"  Error processing protein {protein_id}: {e}")
                # Add dummy prediction to maintain order
                dummy_pred = np.zeros((1, label_network.shape[0]))
                all_predictions.append(dummy_pred)
                continue
    
    return np.vstack(all_predictions), protein_ids

def load_label_network(label_network_file, labels_num):
    """Load label network or create identity matrix"""
    try:
        with open(label_network_file, 'rb') as f:
            label_network = pickle.load(f)
        print(f"  Loaded label network from {label_network_file}")
        return label_network
    except:
        print(f"  Creating identity matrix for {labels_num} labels")
        return torch.eye(labels_num, dtype=torch.float32)

def load_go_mappings(source_data_dir):
    """Load GO term mappings from the source data files"""
    mappings = {}
    
    # Load each ontology's GO terms
    for ontology in ['bp', 'mf', 'cc']:
        go_file = source_data_dir / f'gos_{ontology}.csv'
        print(f"Loading GO terms from {go_file}")
        
        try:
            df = pd.read_csv(go_file)
            
            # Extract unique GO terms for this ontology
            go_column = f'{ontology.upper()}-GO'
            if go_column in df.columns:
                unique_terms = sorted(df[go_column].unique())
            else:
                # Try alternative column names
                possible_cols = [col for col in df.columns if 'GO' in col.upper()]
                if possible_cols:
                    unique_terms = sorted(df[possible_cols[0]].unique())
                else:
                    print(f"Warning: No GO column found in {go_file}")
                    continue
            
            # Create index mapping (term -> index)
            term_to_idx = {term: idx for idx, term in enumerate(unique_terms)}
            idx_to_term = {idx: term for term, idx in term_to_idx.items()}
            
            mappings[ontology.upper()] = {
                'term_to_idx': term_to_idx,
                'idx_to_term': idx_to_term,
                'terms': unique_terms
            }
            
            print(f"  Loaded {len(unique_terms)} {ontology.upper()} terms")
            
        except Exception as e:
            print(f"Error loading {go_file}: {e}")
            continue
    
    return mappings

def load_protein_id_mapping(idmapping_file):
    """Load protein ID mapping from idmapping_uni.txt"""
    print(f"Loading protein ID mappings from {idmapping_file}")
    
    id_mapping = {}
    try:
        # idmapping_uni.txt format: uniprot_id \t other_id
        with open(idmapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    uniprot_id = parts[0]  # e.g., Q6GZX4
                    other_id = parts[1]    # e.g., 001R_FRG3G
                    # Map both directions
                    id_mapping[uniprot_id] = other_id
                    id_mapping[other_id] = uniprot_id
        
        print(f"  Loaded mappings for {len(id_mapping)//2} protein IDs")
        
    except Exception as e:
        print(f"Error loading {idmapping_file}: {e}")
    
    return id_mapping

def map_protein_ids(protein_ids, id_mapping):
    """Map protein IDs using the mapping dictionary"""
    mapped_ids = []
    unmapped_count = 0
    
    for protein_id in protein_ids:
        # Try direct match first
        if protein_id in id_mapping:
            mapped_ids.append(id_mapping[protein_id])
        else:
            # Keep original if no mapping found
            mapped_ids.append(protein_id)
            unmapped_count += 1
    
    if unmapped_count > 0:
        print(f"  Warning: {unmapped_count}/{len(protein_ids)} protein IDs could not be mapped")
    
    return mapped_ids

def combine_predictions(bp_preds, mf_preds, cc_preds, protein_ids, output_dir, go_mappings, threshold=0.5, min_confidence=0.1):
    """Combine predictions from all three ontologies using real GO terms"""
    
    # Create long format data with real GO terms
    long_format_data = []
    
    # Process BP predictions
    if 'BP' in go_mappings:
        bp_terms = go_mappings['BP']['idx_to_term']
        for i, protein_id in enumerate(protein_ids):
            for j in range(bp_preds.shape[1]):
                confidence = bp_preds[i, j]
                if confidence >= min_confidence:
                    go_term = bp_terms.get(j, f"BP_UNKNOWN_{j}")
                    long_format_data.append({
                        'protein_id': protein_id,
                        'go_term': go_term,
                        'ontology': 'BP',
                        'confidence_score': confidence,
                        'predicted': 1 if confidence >= threshold else 0
                    })
    
    # Process MF predictions
    if 'MF' in go_mappings:
        mf_terms = go_mappings['MF']['idx_to_term']
        for i, protein_id in enumerate(protein_ids):
            for j in range(mf_preds.shape[1]):
                confidence = mf_preds[i, j]
                if confidence >= min_confidence:
                    go_term = mf_terms.get(j, f"MF_UNKNOWN_{j}")
                    long_format_data.append({
                        'protein_id': protein_id,
                        'go_term': go_term,
                        'ontology': 'MF',
                        'confidence_score': confidence,
                        'predicted': 1 if confidence >= threshold else 0
                    })
    
    # Process CC predictions
    if 'CC' in go_mappings:
        cc_terms = go_mappings['CC']['idx_to_term']
        for i, protein_id in enumerate(protein_ids):
            for j in range(cc_preds.shape[1]):
                confidence = cc_preds[i, j]
                if confidence >= min_confidence:
                    go_term = cc_terms.get(j, f"CC_UNKNOWN_{j}")
                    long_format_data.append({
                        'protein_id': protein_id,
                        'go_term': go_term,
                        'ontology': 'CC',
                        'confidence_score': confidence,
                        'predicted': 1 if confidence >= threshold else 0
                    })
    
    # Create DataFrame in long format
    long_df = pd.DataFrame(long_format_data)
    
    # Sort by protein_id, then by confidence score (descending)
    long_df = long_df.sort_values(['protein_id', 'confidence_score'], ascending=[True, False])
    
    # Save long format (protein_id - go_term - confidence_score)
    long_output = output_dir / 'predictions_long_format.csv'
    long_df.to_csv(long_output, index=False)
    print(f"Saved long format predictions to {long_output}")
    
    # Create simplified version with just the three columns requested
    simple_df = long_df[['protein_id', 'go_term', 'confidence_score']].copy()
    simple_output = output_dir / 'predictions_simple.csv'
    simple_df.to_csv(simple_output, index=False)
    print(f"Saved simple format (protein_id - go_term - confidence_score) to {simple_output}")
    
    # Create high-confidence predictions only
    high_conf_df = long_df[long_df['confidence_score'] >= threshold].copy()
    high_conf_output = output_dir / 'predictions_high_confidence.csv'
    high_conf_df.to_csv(high_conf_output, index=False)
    print(f"Saved high confidence predictions (>= {threshold}) to {high_conf_output}")
    
    # Create summary statistics
    summary_stats = {
        'ontology': ['BP', 'MF', 'CC', 'Total'],
        'num_terms': [
            len(go_mappings.get('BP', {}).get('terms', [])),
            len(go_mappings.get('MF', {}).get('terms', [])),
            len(go_mappings.get('CC', {}).get('terms', [])),
            len(go_mappings.get('BP', {}).get('terms', [])) + 
            len(go_mappings.get('MF', {}).get('terms', [])) + 
            len(go_mappings.get('CC', {}).get('terms', []))
        ],
        'total_predictions': [
            len(long_df[long_df['ontology'] == 'BP']),
            len(long_df[long_df['ontology'] == 'MF']),
            len(long_df[long_df['ontology'] == 'CC']),
            len(long_df)
        ],
        'high_confidence_predictions': [
            len(high_conf_df[high_conf_df['ontology'] == 'BP']),
            len(high_conf_df[high_conf_df['ontology'] == 'MF']),
            len(high_conf_df[high_conf_df['ontology'] == 'CC']),
            len(high_conf_df)
        ],
        'avg_confidence': [
            long_df[long_df['ontology'] == 'BP']['confidence_score'].mean() if len(long_df[long_df['ontology'] == 'BP']) > 0 else 0,
            long_df[long_df['ontology'] == 'MF']['confidence_score'].mean() if len(long_df[long_df['ontology'] == 'MF']) > 0 else 0,
            long_df[long_df['ontology'] == 'CC']['confidence_score'].mean() if len(long_df[long_df['ontology'] == 'CC']) > 0 else 0,
            long_df['confidence_score'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_output = output_dir / 'prediction_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Saved prediction summary to {summary_output}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Number of proteins: {len(protein_ids)}")
    print(f"Total predictions (>= {min_confidence} confidence): {len(long_df)}")
    print(f"High confidence predictions (>= {threshold}): {len(high_conf_df)}")
    print(f"\nBy ontology:")
    print(f"  BP: {len(long_df[long_df['ontology'] == 'BP'])} predictions ({len(high_conf_df[high_conf_df['ontology'] == 'BP'])} high confidence)")
    print(f"  MF: {len(long_df[long_df['ontology'] == 'MF'])} predictions ({len(high_conf_df[high_conf_df['ontology'] == 'MF'])} high confidence)")
    print(f"  CC: {len(long_df[long_df['ontology'] == 'CC'])} predictions ({len(high_conf_df[high_conf_df['ontology'] == 'CC'])} high confidence)")
    print(f"\nAverage confidence scores:")
    if len(long_df[long_df['ontology'] == 'BP']) > 0:
        print(f"  BP: {long_df[long_df['ontology'] == 'BP']['confidence_score'].mean():.3f}")
    if len(long_df[long_df['ontology'] == 'MF']) > 0:
        print(f"  MF: {long_df[long_df['ontology'] == 'MF']['confidence_score'].mean():.3f}")
    if len(long_df[long_df['ontology'] == 'CC']) > 0:
        print(f"  CC: {long_df[long_df['ontology'] == 'CC']['confidence_score'].mean():.3f}")
    print(f"  Overall: {long_df['confidence_score'].mean():.3f}")
    
    return long_df, high_conf_df

def main():
    parser = argparse.ArgumentParser(description='Multi-ontology Struct2GO prediction')
    parser.add_argument('--processed_data_dir', required=True, help='Directory with processed data (including label networks)')
    parser.add_argument('--models_dir', required=True, help='Directory containing the three model files')
    parser.add_argument('--source_data_dir', required=True, help='Directory containing GO term mappings (gos_bp.csv, etc.)')
    parser.add_argument('--idmapping_file', help='Path to idmapping_uni.txt file (optional)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_confidence', type=float, default=0.1, help='Minimum confidence to include in output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    processed_data_dir = Path(args.processed_data_dir)
    output_dir = Path(args.output_dir)
    source_data_dir = Path(args.source_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations (after processed_data_dir is defined)
    model_configs = {
        'BP': {
            'model_file': 'mymodel_bp_1_0.0005_0.45.pkl',
            'labels_num': 809,
            'label_network_file': processed_data_dir / 'label_bp_network.pkl'
        },
        'MF': {
            'model_file': 'mymodel_mf_1_0.0005_0.45.pkl', 
            'labels_num': 273,
            'label_network_file': processed_data_dir / 'label_mf_network.pkl'
        },
        'CC': {
            'model_file': 'mymodel_cc_1_0.0005_0.45.pkl',
            'labels_num': 298,
            'label_network_file': processed_data_dir / 'label_cc_network.pkl'
        }
    }
    
    # Load GO term mappings
    go_mappings = load_go_mappings(source_data_dir)
    
    # Load protein ID mappings if provided
    id_mapping = {}
    if args.idmapping_file and Path(args.idmapping_file).exists():
        id_mapping = load_protein_id_mapping(args.idmapping_file)
    
    print("Loading data files...")
    
    # Load graphs from processed data directory
    graphs_file = processed_data_dir / 'emb_graph_test.pkl'
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} protein graphs")
    
    # Load node features
    node_features_file = processed_data_dir / 'protein_node2onehot.pkl'
    with open(node_features_file, 'rb') as f:
        node_features = pickle.load(f)
    print(f"Loaded node features for {len(node_features)} proteins")
    
    # Load sequence features
    sequence_features_file = processed_data_dir / 'dict_sequence_feature.pkl'
    with open(sequence_features_file, 'rb') as f:
        sequence_features = pickle.load(f)
    print(f"Loaded sequence features for {len(sequence_features)} proteins")
    
    # Get protein IDs and map them if needed
    original_protein_ids = list(graphs.keys())
    if id_mapping:
        print("Mapping protein IDs...")
        mapped_protein_ids = map_protein_ids(original_protein_ids, id_mapping)
        print(f"  Original IDs (sample): {original_protein_ids[:5]}")
        print(f"  Mapped IDs (sample): {mapped_protein_ids[:5]}")
    else:
        mapped_protein_ids = original_protein_ids
    
    # Verify data consistency
    common_proteins = set(graphs.keys()) & set(node_features.keys()) & set(sequence_features.keys())
    print(f"Common proteins across all data: {len(common_proteins)}")
    
    if len(common_proteins) == 0:
        print("ERROR: No common proteins found!")
        return
    
    # Filter to common proteins
    graphs = {p: graphs[p] for p in common_proteins}
    node_features = {p: node_features[p] for p in common_proteins}
    sequence_features = {p: sequence_features[p] for p in common_proteins}
    
    # Store predictions
    all_predictions = {}
    
    # Process each ontology
    for ontology, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Processing {ontology} ontology")
        print(f"{'='*50}")
        
        # Load model
        model_path = models_dir / config['model_file']
        print(f"Loading model: {model_path}")
        
        try:
            model = torch.load(model_path, map_location=args.device)
            model.to(args.device)
            model.eval()
            
            # Fix missing label_network1 attribute if needed
            if not hasattr(model, 'label_network1'):
                print(f"  Adding missing label_network1 to {ontology} model...")
                from dgl.nn.pytorch.conv import GATConv
                model.label_network1 = GATConv(1, 1, num_heads=8, allow_zero_in_degree=True).to(args.device)
            
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        # Load/create label network
        label_network = load_label_network(config['label_network_file'], config['labels_num'])
        
        # Run predictions
        predictions, protein_ids = predict_ontology(
            model, graphs, node_features, sequence_features, 
            label_network, args.device, ontology
        )
        
        all_predictions[ontology] = predictions
        print(f"  {ontology} predictions completed: {predictions.shape}")
    
    # Use mapped protein IDs for final output
    final_protein_ids = mapped_protein_ids if id_mapping else protein_ids
    
    # Combine all predictions
    if len(all_predictions) == 3:
        print(f"\n{'='*50}")
        print("Combining predictions from all ontologies...")
        print(f"{'='*50}")
        
        combined_df, binary_df = combine_predictions(
            all_predictions['BP'],
            all_predictions['MF'], 
            all_predictions['CC'],
            final_protein_ids,
            output_dir,
            go_mappings,
            args.threshold,
            args.min_confidence
        )
        
        print(f"\nAll predictions saved to {output_dir}")
        print("Files created:")
        print("  - predictions_simple.csv (protein_id - go_term - confidence_score)")
        print("  - predictions_long_format.csv (includes ontology and binary prediction)")
        print("  - predictions_high_confidence.csv (only high confidence predictions)")
        print("  - prediction_summary.csv (summary statistics)")
        
    else:
        print(f"Warning: Only {len(all_predictions)} ontologies processed successfully")

if __name__ == "__main__":
    main()