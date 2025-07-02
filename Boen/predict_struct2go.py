import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance
import warnings
import datetime
import pandas as pd
import os
from pathlib import Path

warnings.filterwarnings('ignore')

def load_go_term_mapping(mapping_file):
    """Load GO term mapping from pickle file"""
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    return mapping['idx_to_term'], mapping['term_to_idx']

def create_prediction_tsv(predictions, protein_ids, go_mapping, branch, confidence_threshold=0.1):
    """
    Create prediction results for one branch
    
    Args:
        predictions: numpy array of prediction scores [n_proteins, n_go_terms]
        protein_ids: list of protein IDs
        go_mapping: dict mapping index to GO term
        branch: GO branch name (mf, bp, cc)
        confidence_threshold: minimum confidence score to include
    """
    results = []
    
    for i, protein_id in enumerate(protein_ids):
        protein_predictions = predictions[i]
        
        # Get indices of GO terms above threshold
        significant_indices = np.where(protein_predictions >= confidence_threshold)[0]
        
        for go_idx in significant_indices:
            go_term = go_mapping.get(go_idx, f"GO_UNKNOWN_{go_idx}")
            confidence = float(protein_predictions[go_idx])
            
            results.append({
                'protein_id': protein_id,
                'go_term': go_term,
                'confidence_score': confidence
            })
    
    return results

def combine_and_save_predictions(all_results, output_file):
    """Combine all branch predictions and save to TSV"""
    # Create DataFrame from all results
    df = pd.DataFrame(all_results)
    
    # Sort by protein_id, go_branch, and confidence_score (descending)
    df = df.sort_values(['protein_id', 'go_branch', 'confidence_score'], 
                       ascending=[True, True, False])
    
    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n=== FINAL COMBINED RESULTS ===")
    print(f"Total predictions: {len(df)}")
    print(f"Unique proteins: {len(df['protein_id'].unique())}")
    print(f"Unique GO terms: {len(df['go_term'].unique())}")
    
    # Branch-wise summary
    for branch in ['MF', 'BP', 'CC']:
        branch_df = df[df['go_branch'] == branch]
        if len(branch_df) > 0:
            print(f"{branch}: {len(branch_df)} predictions, "
                  f"avg confidence: {branch_df['confidence_score'].mean():.4f}")
    
    return df

def extract_protein_ids_from_dataset(dataset):
    """Extract protein IDs from dataset - you may need to modify this based on your dataset structure"""
    # This is a placeholder - you'll need to adjust based on how protein IDs are stored in your dataset
    protein_ids = []
    
    if hasattr(dataset, 'protein_ids'):
        protein_ids = dataset.protein_ids
    elif hasattr(dataset, 'graphs'):
        # If protein IDs are stored as graph attributes
        for graph in dataset.graphs:
            if hasattr(graph, 'protein_id'):
                protein_ids.append(graph.protein_id)
            else:
                protein_ids.append(f"protein_{len(protein_ids)}")
    else:
        # Fallback: generate generic IDs
        protein_ids = [f"protein_{i}" for i in range(len(dataset))]
    
    return protein_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input data and models
    parser.add_argument('-data_dir', '--data_dir', type=str, required=True,
                       help='Directory containing processed dataset files (emb_graph_*.pkl, etc.)')
    parser.add_argument('-model_dir', '--model_dir', type=str, required=True,
                       help='Directory containing trained model files')
    parser.add_argument('-model_mf', '--model_mf', type=str, default='model_mf.pkl',
                       help='MF model filename')
    parser.add_argument('-model_bp', '--model_bp', type=str, default='model_bp.pkl',
                       help='BP model filename')
    parser.add_argument('-model_cc', '--model_cc', type=str, default='model_cc.pkl',
                       help='CC model filename')
    
    # Optional parameters
    parser.add_argument('-confidence_threshold', '--confidence_threshold', type=float, default=0.1,
                       help='Minimum confidence score to include in output')
    parser.add_argument('-output_file', '--output_file', type=str, default=None,
                       help='Output TSV file path')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Set output file if not provided
    if args.output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"struct2go_predictions_all_branches_{timestamp}.tsv"
    
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    
    print("=== Struct2GO Multi-Branch Prediction ===")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define branch configurations
    branches = {
        'mf': {
            'data_files': ['emb_graph_mf.pkl', 'emb_seq_feature_mf.pkl'],
            'mapping_file': 'go_term_mapping_molecular_function.pkl',
            'model_file': args.model_mf,
            'default_labels': 273
        },
        'bp': {
            'data_files': ['emb_graph_bp.pkl', 'emb_seq_feature_bp.pkl'],
            'mapping_file': 'go_term_mapping_biological_process.pkl',
            'model_file': args.model_bp,
            'default_labels': 1000
        },
        'cc': {
            'data_files': ['emb_graph_cc.pkl', 'emb_seq_feature_cc.pkl'],
            'mapping_file': 'go_term_mapping_cellular_component.pkl',
            'model_file': args.model_cc,
            'default_labels': 200
        }
    }
    
    all_results = []
    
    for branch, config in branches.items():
        print(f"\n=== Processing {branch.upper()} Branch ===")
        
        # Check if required files exist
        graph_file = data_dir / config['data_files'][0]
        feature_file = data_dir / config['data_files'][1]
        mapping_file = data_dir / config['mapping_file']
        model_file = model_dir / config['model_file']
        
        missing_files = []
        if not graph_file.exists():
            missing_files.append(str(graph_file))
        if not feature_file.exists():
            missing_files.append(str(feature_file))
        if not mapping_file.exists():
            missing_files.append(str(mapping_file))
        if not model_file.exists():
            missing_files.append(str(model_file))
        
        if missing_files:
            print(f"Skipping {branch.upper()} - missing files:")
            for f in missing_files:
                print(f"  - {f}")
            continue
        
        try:
            # Load data
            print(f"Loading {branch.upper()} data...")
            with open(graph_file, 'rb') as f:
                graphs = pickle.load(f)
            with open(feature_file, 'rb') as f:
                features = pickle.load(f)
            
            # Load GO mapping
            print(f"Loading {branch.upper()} GO mapping...")
            idx_to_go, go_to_idx = load_go_term_mapping(mapping_file)
            
            # Load model
            print(f"Loading {branch.upper()} model...")
            model = torch.load(model_file, map_location=device)
            model.eval()
            
            # Get protein IDs (assuming they're the keys in the graphs dict)
            protein_ids = list(graphs.keys())
            print(f"Found {len(protein_ids)} proteins")
            
            # Create dataset from loaded data
            # You may need to adapt this based on your actual dataset structure
            dataset_items = []
            for protein_id in protein_ids:
                if protein_id in graphs and protein_id in features:
                    dataset_items.append((graphs[protein_id], features[protein_id]))
            
            print(f"Created dataset with {len(dataset_items)} items")
            
            # Run predictions
            print(f"Running {branch.upper()} predictions...")
            all_predictions = []
            
            with torch.no_grad():
                for i, (graph, feature) in enumerate(dataset_items):
                    # Move to device
                    if hasattr(graph, 'to'):
                        graph = graph.to(device)
                    feature = feature.to(device)
                    
                    # Add batch dimension if needed
                    if len(feature.shape) == 2:
                        feature = feature.unsqueeze(0)
                    
                    try:
                        # Try prediction (model signature may vary)
                        logits = model(graph, feature)
                    except:
                        try:
                            # Some models might need label network
                            dummy_label_network = torch.eye(len(idx_to_go)).to(device)
                            logits = model(graph, feature, dummy_label_network)
                        except Exception as e:
                            print(f"Error predicting for protein {protein_ids[i]}: {e}")
                            continue
                    
                    # Apply sigmoid and store
                    pred = torch.sigmoid(logits).cpu().numpy()
                    if len(pred.shape) > 1:
                        pred = pred[0]  # Remove batch dimension
                    all_predictions.append(pred)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(dataset_items)} proteins...")
            
            if len(all_predictions) == 0:
                print(f"No successful predictions for {branch.upper()}")
                continue
                
            # Convert to numpy
            all_predictions = np.array(all_predictions)
            successful_protein_ids = protein_ids[:len(all_predictions)]
            
            print(f"Generated predictions for {len(successful_protein_ids)} proteins")
            print(f"Prediction matrix shape: {all_predictions.shape}")
            
            # Create results for this branch
            branch_results = create_prediction_tsv(
                predictions=all_predictions,
                protein_ids=successful_protein_ids,
                go_mapping=idx_to_go,
                branch=branch,
                confidence_threshold=args.confidence_threshold
            )
            
            all_results.extend(branch_results)
            print(f"{branch.upper()} branch: {len(branch_results)} predictions above threshold")
            
        except Exception as e:
            print(f"Error processing {branch.upper()} branch: {e}")
            continue
    
    if not all_results:
        print("No predictions generated for any branch!")
        exit(1)
    
    # Combine and save all results
    print(f"\nCombining results from all branches...")
    final_df = combine_and_save_predictions(all_results, args.output_file)
    
    # Show sample of final results
    print(f"\n=== SAMPLE PREDICTIONS ===")
    print("protein_id\tgo_term\tgo_branch\tconfidence_score")
    for _, row in final_df.head(10).iterrows():
        print(f"{row['protein_id']}\t{row['go_term']}\t{row['go_branch']}\t{row['confidence_score']:.4f}")
    
    print(f"\nAll predictions saved to: {args.output_file}")
    print("Done!")