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

def create_legacy_dataset(graphs, features, labels_num=273):
    """Create dataset compatible with older DGL versions"""
    dataset = []
    protein_ids = list(graphs.keys())
    
    for protein_id in protein_ids:
        graph = graphs[protein_id]
        feature = features[protein_id]
        dummy_labels = torch.zeros(labels_num)
        
        # For older DGL, ensure graph is properly formatted
        if hasattr(graph, 'ndata'):
            # Add features to graph nodes (legacy format)
            graph.ndata['feat'] = feature  # Try 'feat' instead of 'feature'
            graph.ndata['h'] = feature     # Also try 'h' as backup
        
        # Legacy dataset format
        dataset.append((graph, dummy_labels, feature))
    
    return dataset, protein_ids

def predict_legacy(model, graphs, features, label_network, device='cpu'):
    """Legacy prediction without DataLoader issues"""
    
    model.eval()
    all_predictions = []
    protein_ids = list(graphs.keys())
    
    print(f"Running legacy predictions on {len(protein_ids)} proteins...")
    
    with torch.no_grad():
        for i, protein_id in enumerate(protein_ids):
            try:
                graph = graphs[protein_id].to(device)
                feature = features[protein_id].to(device)
                
                # Ensure graph has node features
                if 'feat' not in graph.ndata and 'feature' not in graph.ndata:
                    graph.ndata['feat'] = feature
                    graph.ndata['feature'] = feature
                
                # Create batch of 1
                batched_graph = dgl.batch([graph])
                batched_feature = feature.unsqueeze(0)  # Add batch dimension
                
                # Model prediction
                logits = model(batched_graph, batched_feature, label_network.to(device))
                predictions = torch.sigmoid(logits)
                
                all_predictions.append(predictions.cpu().numpy())
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(protein_ids)} proteins...")
                    
            except Exception as e:
                print(f"Error processing protein {protein_id}: {e}")
                # Add dummy prediction to maintain order
                dummy_pred = np.zeros((1, label_network.shape[0]))
                all_predictions.append(dummy_pred)
                continue
    
    return np.vstack(all_predictions), protein_ids

def save_predictions(predictions, protein_ids, output_path, threshold=0.5):
    """Save predictions to file"""
    
    go_terms = [f"GO_term_{i}" for i in range(predictions.shape[1])]
    
    # Raw predictions
    df = pd.DataFrame(predictions, columns=go_terms, index=protein_ids)
    df.index.name = 'protein_id'
    
    raw_output = output_path.replace('.csv', '_raw_predictions.csv')
    df.to_csv(raw_output)
    print(f"Saved raw predictions to {raw_output}")
    
    # Binary predictions
    binary_predictions = (predictions > threshold).astype(int)
    df_binary = pd.DataFrame(binary_predictions, columns=go_terms, index=protein_ids)
    df_binary.index.name = 'protein_id'
    binary_output = output_path.replace('.csv', '_binary_predictions.csv')
    df_binary.to_csv(binary_output)
    print(f"Saved binary predictions to {binary_output}")
    
    print(f"\nPrediction Summary:")
    print(f"Number of proteins: {len(protein_ids)}")
    print(f"Number of GO terms: {predictions.shape[1]}")
    print(f"Average predictions per protein: {np.mean(np.sum(binary_predictions, axis=1)):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Legacy prediction for older DGL/PyTorch')
    parser.add_argument('--graphs_file', required=True)
    parser.add_argument('--features_file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--labels_num', type=int, default=273)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading structural data...")
    with open(args.graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    
    with open(args.features_file, 'rb') as f:
        features = pickle.load(f)
    
    print(f"Loaded data for {len(graphs)} proteins")
    
    # Simple label network
    label_network = torch.eye(args.labels_num, dtype=torch.float32)
    
    # Load model with legacy compatibility
    print("Loading model...")
    try:
        model = torch.load(args.model, map_location=args.device)
        model.to(args.device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run legacy predictions
    predictions, protein_ids = predict_legacy(model, graphs, features, label_network, args.device)
    
    # Save results
    output_path = output_dir / 'predictions.csv'
    save_predictions(predictions, protein_ids, str(output_path), args.threshold)
    
    print(f"\nDone! Check {output_dir} for results.")

if __name__ == "__main__":
    main()