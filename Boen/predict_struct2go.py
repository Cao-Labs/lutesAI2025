import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_dataset_from_structural_data(graphs, features, labels_num=273):
    """Create dataset from your structural data with dummy labels"""
    dataset = []
    protein_ids = list(graphs.keys())
    
    for protein_id in protein_ids:
        graph = graphs[protein_id]
        feature = features[protein_id]
        dummy_labels = torch.zeros(labels_num)  # Dummy labels, not used for prediction
        
        # Format: (graph, labels, sequence_feature) - matches original script
        dataset.append((graph, dummy_labels, feature))
    
    return dataset, protein_ids

def predict_only(model, test_dataloader, label_network, device='cuda'):
    """Run predictions without evaluation"""
    
    model.eval()
    all_predictions = []
    batch_count = 0
    
    print("Running predictions...")
    
    with torch.no_grad():
        for batched_graph, labels, sequence_feature in test_dataloader:
            # Same prediction logic as original script
            logits = model(batched_graph.to(device), sequence_feature.to(device), label_network.to(device))
            predictions = torch.sigmoid(logits)
            
            all_predictions.append(predictions.cpu().numpy())
            batch_count += 1
            
            if batch_count % 100 == 0:
                print(f"Processed {batch_count} batches...")
    
    return np.vstack(all_predictions)

def save_predictions(predictions, protein_ids, output_path, threshold=0.5):
    """Save predictions to file"""
    
    # Create simple column names
    go_terms = [f"GO_term_{i}" for i in range(predictions.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=go_terms, index=protein_ids)
    df.index.name = 'protein_id'
    
    # Save raw predictions
    raw_output = output_path.replace('.csv', '_raw_predictions.csv')
    df.to_csv(raw_output)
    print(f"Saved raw predictions to {raw_output}")
    
    # Save thresholded predictions
    binary_predictions = (predictions > threshold).astype(int)
    df_binary = pd.DataFrame(binary_predictions, columns=go_terms, index=protein_ids)
    df_binary.index.name = 'protein_id'
    binary_output = output_path.replace('.csv', '_binary_predictions.csv')
    df_binary.to_csv(binary_output)
    print(f"Saved binary predictions (threshold={threshold}) to {binary_output}")
    
    # Summary
    print(f"\nPrediction Summary:")
    print(f"Number of proteins: {len(protein_ids)}")
    print(f"Number of GO terms: {predictions.shape[1]}")
    print(f"Average predictions per protein: {np.mean(np.sum(binary_predictions, axis=1)):.2f}")
    print(f"Score range: {np.min(predictions):.3f} - {np.max(predictions):.3f}")

def main():
    parser = argparse.ArgumentParser(description='Run predictions from structural data only')
    parser.add_argument('--graphs_file', required=True, help='Path to protein_graphs.pkl')
    parser.add_argument('--features_file', required=True, help='Path to protein_node_features.pkl')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--labels_num', type=int, default=273, help='Number of labels model expects')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading structural data...")
    
    # Load your processed structural data
    with open(args.graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    
    with open(args.features_file, 'rb') as f:
        features = pickle.load(f)
    
    print(f"Loaded data for {len(graphs)} proteins")
    
    # Create simple label network (identity matrix)
    label_network = torch.eye(args.labels_num, dtype=torch.float32)
    
    # Create dataset
    dataset, protein_ids = create_dataset_from_structural_data(graphs, features, args.labels_num)
    
    # Create dataloader
    dataloader = GraphDataLoader(
        dataset=dataset, 
        batch_size=1,
        drop_last=False, 
        shuffle=False
    )
    
    # Load model
    print("Loading model...")
    model = torch.load(args.model)
    model.to(args.device)
    
    # Run predictions
    predictions = predict_only(model, dataloader, label_network, args.device)
    
    # Save results
    output_path = output_dir / 'predictions.csv'
    save_predictions(predictions, protein_ids, str(output_path), args.threshold)
    
    print(f"\nDone! Check {output_dir} for results.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()