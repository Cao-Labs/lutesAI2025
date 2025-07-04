import os
import torch
import argparse
from src import Gnn_PF_fixed, data_loader_fixed
from torch.utils import data as D
from torch_geometric.data import DataLoader

current_file_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = current_file_path + '/data/'

def predict(model, loader, device):
    results = dict()
    for data in loader:
        # Run on CPU to avoid CUDA issues
        esm_rep = data.x.T.unsqueeze(0).cpu()     # [1, 1280, seq_len]
        seq = data.seq.T.unsqueeze(0).cpu()       # [1, 25, seq_len]  
        contact = data.edge_index.cpu()
        pssm = data.pssm.T.unsqueeze(0).cpu()     # [1, 20, seq_len]
        seq_embed = data.seq_embed.cpu()
        label = data.label
        batch_idx = data.batch.cpu()
        
        model_pred = torch.sigmoid(model(
            esm_rep=esm_rep, 
            seq=seq, 
            pssm=pssm, 
            seq_embed=seq_embed, 
            A=contact, 
            batch=batch_idx
        )).cpu().detach().numpy()
        
        for i, chain_id in enumerate(data.chain_id):
            results[chain_id] = model_pred[i, :]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting Protein function with GAT-GO')
    parser.add_argument('--ModelPath', help='Model to be used for inference', type=str, required=True)
    parser.add_argument('--Device', help='CUDA device for inference', type=int, default=0)
    parser.add_argument('--BatchSize', help='Batch size for inference', type=int, default=4)
    parser.add_argument('--SeqIDs', help='Input seq file for inference', type=str, required=True)
    parser.add_argument('--OutDir', help='Output Directory to store result', type=str, required=True)
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.SeqIDs):
        print('Error: Input file does not exist')
        exit(1)
        
    # Check if model file exists
    if not os.path.isfile(args.ModelPath):
        print('Error: Model file does not exist')
        exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(args.OutDir):
        os.makedirs(args.OutDir, exist_ok=True)
    
    # Load dataset
    Dset = data_loader_fixed.Protein_Gnn_data(
        root=DATA_PATH, 
        chain_list=args.SeqIDs
    )
    loader = DataLoader(Dset, batch_size=args.BatchSize)
    
    # Set device to CPU
    device = torch.device('cpu')
    
    # Load model on CPU
    check_point = torch.load(args.ModelPath, map_location=device)
    model = Gnn_PF_fixed.GnnPF().cpu()  # Load model on CPU
    model.load_state_dict(check_point['state_dict'])
    model.eval()
    
    # Run predictions
    results = predict(model, loader, args.Device)
    
    # Save results
    output_file = os.path.join(args.OutDir, 'GAT-GO_Results.pt')
    torch.save(results, output_file)
    
    print(f"Predictions saved to: {output_file}")
    print(f"Number of proteins processed: {len(results)}")