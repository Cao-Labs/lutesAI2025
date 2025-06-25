import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from deeppfp_model import Net
from representationsDataset import representationsDataset

def run_predictions(test_path, state_path, batch_size=64, layer=48, nums=None, device=torch.device("cpu")):
    # Load testing dataset
    test_dataset = representationsDataset(test_path, layer, nums)
    test_dataloader = DataLoader(
        batch_size=batch_size,
        dataset=test_dataset,
        shuffle=False,
        drop_last=False
    )

    # Initialize model and load trained weights
    model = Net(16)
    print('Loading model weights from:', state_path)
    model.load_state_dict(torch.load(state_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for data, _ in test_dataloader:  # Ignoring labels
            data = data.to(device)
            predictions = model(data)
            all_preds.extend(predictions.cpu().numpy())

    # Save predictions to the same directory as the test file
    test_dir = os.path.dirname(test_path)
    output_file = os.path.join(test_dir, "predictions.npy")
    np.save(output_file, np.array(all_preds))
    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on test FASTA data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to testing FASTA file')
    parser.add_argument('--state_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--layer', type=int, default=48, help='ESM layer to use')
    parser.add_argument('--nums', type=int, default=None, help='Optional number of samples to read')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cpu" or "cuda"')

    args = parser.parse_args()

    run_predictions(
        test_path=args.test_path,
        state_path=args.state_path,
        batch_size=args.batch_size,
        layer=args.layer,
        nums=args.nums,
        device=torch.device(args.device)
    )
