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

def combine_predictions(bp_preds, mf_preds, cc_preds, protein_ids, output_dir, threshold=0.5, min_confidence=0.1):
    """Combine predictions from all three ontologies in long format"""
    
    # Create column names for each ontology
    bp_cols = [f"BP_GO_{i:04d}" for i in range(bp_preds.shape[1])]
    mf_cols = [f"MF_GO_{i:04d}" for i in range(mf_preds.shape[1])]
    cc_cols = [f"CC_GO_{i:04d}" for i in range(cc_preds.shape[1])]
    
    # Create long format data
    long_format_data = []
    
    # Process BP predictions
    for i, protein_id in enumerate(protein_ids):
        for j, go_term in enumerate(bp_cols):
            confidence = bp_preds[i, j]
            if confidence >= min_confidence:  # Only include predictions above minimum confidence
                long_format_data.append({
                    'protein_id': protein_id,
                    'go_term': go_term,
                    'ontology': 'BP',
                    'confidence_score': confidence,
                    'predicted': 1 if confidence >= threshold else 0
                })
    
    # Process MF predictions
    for i, protein_id in enumerate(protein_ids):
        for j, go_term in enumerate(mf_cols):
            confidence = mf_preds[i, j]
            if confidence >= min_confidence:
                long_format_data.append({
                    'protein_id': protein_id,
                    'go_term': go_term,
                    'ontology': 'MF',
                    'confidence_score': confidence,
                    'predicted': 1 if confidence >= threshold else 0
                })
    
    # Process CC predictions
    for i, protein_id in enumerate(protein_ids):
        for j, go_term in enumerate(cc_cols):
            confidence = cc_preds[i, j]
            if confidence >= min_confidence:
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
    
    # Also save traditional wide format for compatibility
    bp_df = pd.DataFrame(bp_preds, columns=bp_cols, index=protein_ids)
    mf_df = pd.DataFrame(mf_preds, columns=mf_cols, index=protein_ids)
    cc_df = pd.DataFrame(cc_preds, columns=cc_cols, index=protein_ids)
    
    combined_df = pd.concat([bp_df, mf_df, cc_df], axis=1)
    combined_df.index.name = 'protein_id'
    
    wide_output = output_dir / 'predictions_wide_format.csv'
    combined_df.to_csv(wide_output)
    print(f"Saved wide format predictions to {wide_output}")
    
    # Create summary statistics
    summary_stats = {
        'ontology': ['BP', 'MF', 'CC', 'Total'],
        'num_terms': [bp_preds.shape[1], mf_preds.shape[1], cc_preds.shape[1], 
                     bp_preds.shape[1] + mf_preds.shape[1] + cc_preds.shape[1]],
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
            long_df[long_df['ontology'] == 'BP']['confidence_score'].mean(),
            long_df[long_df['ontology'] == 'MF']['confidence_score'].mean(),
            long_df[long_df['ontology'] == 'CC']['confidence_score'].mean(),
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
    print(f"  BP: {long_df[long_df['ontology'] == 'BP']['confidence_score'].mean():.3f}")
    print(f"  MF: {long_df[long_df['ontology'] == 'MF']['confidence_score'].mean():.3f}")
    print(f"  CC: {long_df[long_df['ontology'] == 'CC']['confidence_score'].mean():.3f}")
    print(f"  Overall: {long_df['confidence_score'].mean():.3f}")
    
    return long_df, high_conf_df

def main():
    parser = argparse.ArgumentParser(description='Multi-ontology Struct2GO prediction')
    parser.add_argument('--graphs_file', required=True, help='Path to protein graphs')
    parser.add_argument('--node_features_file', required=True, help='Path to 56-dim node features')
    parser.add_argument('--sequence_features_file', required=True, help='Path to 1024-dim sequence features')
    parser.add_argument('--models_dir', required=True, help='Directory containing the three model files')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_confidence', type=float, default=0.1, help='Minimum confidence to include in output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations
    model_configs = {
        'BP': {
            'model_file': 'mymodel_bp_1_0.0005_0.45.pkl',
            'labels_num': 809,
            'label_network': None  # Will use identity matrix
        },
        'MF': {
            'model_file': 'mymodel_mf_1_0.0005_0.45.pkl', 
            'labels_num': 273,
            'label_network': None
        },
        'CC': {
            'model_file': 'mymodel_cc_1_0.0005_0.45.pkl',
            'labels_num': 298,
            'label_network': None
        }
    }
    
    print("Loading data files...")
    
    # Load graphs
    with open(args.graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} protein graphs")
    
    # Load node features
    with open(args.node_features_file, 'rb') as f:
        node_features = pickle.load(f)
    print(f"Loaded node features for {len(node_features)} proteins")
    
    # Load sequence features
    with open(args.sequence_features_file, 'rb') as f:
        sequence_features = pickle.load(f)
    print(f"Loaded sequence features for {len(sequence_features)} proteins")
    
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
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        # Load/create label network
        label_network = load_label_network(config['label_network'], config['labels_num'])
        
        # Run predictions
        predictions, protein_ids = predict_ontology(
            model, graphs, node_features, sequence_features, 
            label_network, args.device, ontology
        )
        
        all_predictions[ontology] = predictions
        print(f"  {ontology} predictions completed: {predictions.shape}")
    
    # Combine all predictions
    if len(all_predictions) == 3:
        print(f"\n{'='*50}")
        print("Combining predictions from all ontologies...")
        print(f"{'='*50}")
        
        combined_df, binary_df = combine_predictions(
            all_predictions['BP'],
            all_predictions['MF'], 
            all_predictions['CC'],
            protein_ids,
            output_dir,
            args.threshold,
            args.min_confidence
        )
        
        print(f"\nAll predictions saved to {output_dir}")
        print("Files created:")
        print("  - predictions_simple.csv (protein_id - go_term - confidence_score)")
        print("  - predictions_long_format.csv (includes ontology and binary prediction)")
        print("  - predictions_high_confidence.csv (only high confidence predictions)")
        print("  - predictions_wide_format.csv (traditional matrix format)")
        print("  - prediction_summary.csv (summary statistics)")
        
    else:
        print(f"Warning: Only {len(all_predictions)} ontologies processed successfully")

if __name__ == "__main__":
    main()