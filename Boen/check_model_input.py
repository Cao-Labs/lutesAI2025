#!/usr/bin/env python3
"""
Check what input dimensions the trained model expects
"""

import torch
import pickle
from pathlib import Path

def check_model_architecture(model_path):
    """Load model and inspect its architecture"""
    
    print(f"Loading model from {model_path}")
    
    try:
        model = torch.load(model_path, map_location='cpu')
        print("Model loaded successfully!")
        
        # Print model architecture
        print("\nModel architecture:")
        print(model)
        
        # Try to find input layer dimensions
        for name, param in model.named_parameters():
            if 'conv' in name.lower() or 'linear' in name.lower() or 'embed' in name.lower():
                print(f"\nLayer: {name}")
                print(f"Shape: {param.shape}")
                
        # Look for the first layer that processes node features
        if hasattr(model, 'convpools') and len(model.convpools) > 0:
            first_conv = model.convpools[0]
            print(f"\nFirst conv layer: {first_conv}")
            
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def check_existing_struct2go_data():
    """Check if there are existing Struct2GO data files to see expected format"""
    
    # Look for existing data files in the Struct2GO directory
    struct2go_dir = Path("/data/shared/tools/Struct2GO")
    
    # Common data file patterns
    data_patterns = [
        "processed_data/*.pkl",
        "divided_data/*.pkl", 
        "data/*.pkl",
        "**/emb_*.pkl",
        "**/test_*.pkl"
    ]
    
    print("Looking for existing Struct2GO data files...")
    
    for pattern in data_patterns:
        files = list(struct2go_dir.glob(pattern))
        if files:
            print(f"\nFound files matching {pattern}:")
            for file in files[:5]:  # Show first 5
                print(f"  {file}")
                
                # Try to load and check dimensions
                try:
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        sample_key = next(iter(data.keys()))
                        sample_data = data[sample_key]
                        if hasattr(sample_data, 'shape'):
                            print(f"    Sample shape: {sample_data.shape}")
                        elif isinstance(sample_data, torch.Tensor):
                            print(f"    Sample tensor shape: {sample_data.shape}")
                            
                except Exception as e:
                    print(f"    Error loading {file}: {e}")

def main():
    # Check model architecture
    model_path = "/data/shared/tools/Struct2GO/save_models/mymodel_mf_1_0.0005_0.45.pkl"
    model = check_model_architecture(model_path)
    
    # Check existing data
    check_existing_struct2go_data()
    
    # Print recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("1. Check if there are existing processed data files in /data/shared/tools/Struct2GO/")
    print("2. The model expects 56-dimensional node features")
    print("3. You may need to reprocess your data with the correct feature dimension")
    print("4. Or find the original training data format and match that")

if __name__ == "__main__":
    main()