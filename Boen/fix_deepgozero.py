#!/usr/bin/env python3
"""
Systematic DeepGOZero debugging and fixing script
"""

import torch
import os
import sys
sys.path.append('/data/shared/tools/deepgozero')

def check_pytorch_version():
    """Check PyTorch version compatibility"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
def check_model_files():
    """Check all available model files and their contents"""
    data_dir = "/data/shared/tools/deepgozero/data/bp"
    
    print("\n=== CHECKING MODEL FILES ===")
    
    model_files = [f for f in os.listdir(data_dir) if f.endswith('.th')]
    print(f"Available model files: {model_files}")
    
    for model_file in model_files:
        model_path = os.path.join(data_dir, model_file)
        print(f"\n--- {model_file} ---")
        try:
            # Try loading with different methods
            model_data = torch.load(model_path, map_location='cpu')
            
            if isinstance(model_data, dict):
                print(f"Keys in model: {list(model_data.keys())[:10]}")
                
                # Check for state_dict
                if 'state_dict' in model_data:
                    print("Model has 'state_dict' key")
                    actual_state = model_data['state_dict']
                else:
                    actual_state = model_data
                
                # Look for the problematic keys
                missing_keys = []
                for key in actual_state.keys():
                    if 'layer_norm' in key:
                        print(f"Found layer_norm key: {key}")
                
                # Check if we need to strip prefixes
                sample_keys = list(actual_state.keys())[:5]
                print(f"Sample keys: {sample_keys}")
                
            else:
                print(f"Model type: {type(model_data)}")
                
        except Exception as e:
            print(f"Error loading {model_file}: {e}")

def try_load_with_different_methods():
    """Try different loading methods for the model"""
    print("\n=== TRYING DIFFERENT LOADING METHODS ===")
    
    model_path = "/data/shared/tools/deepgozero/data/bp/deepgozero.th"
    
    # Method 1: Direct load
    try:
        print("Method 1: Direct torch.load")
        model = torch.load(model_path, map_location='cpu')
        print("✅ Direct load successful")
        return model
    except Exception as e:
        print(f"❌ Direct load failed: {e}")
    
    # Method 2: Load with strict=False
    try:
        print("Method 2: Load with strict=False")
        # We'll need to create the model first
        # This requires knowing the architecture
        print("Need to create model architecture first...")
    except Exception as e:
        print(f"❌ strict=False failed: {e}")
    
    # Method 3: Try different model files
    for model_name in ['deepgozero_zero.th', 'deepgozero_zero_10.th']:
        try:
            print(f"Method 3: Trying {model_name}")
            alt_path = f"/data/shared/tools/deepgozero/data/bp/{model_name}"
            if os.path.exists(alt_path):
                model = torch.load(alt_path, map_location='cpu')
                print(f"✅ {model_name} loaded successfully")
                return model
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    return None

def check_deepgozero_architecture():
    """Check what architecture DeepGOZero expects"""
    print("\n=== CHECKING DEEPGOZERO ARCHITECTURE ===")
    
    try:
        from deepgozero import DGELModel
        
        # Try to create a model with dummy parameters to see expected architecture
        print("Attempting to create DGELModel...")
        
        # These are dummy values - we need the real ones from the data files
        num_interpros = 100  # dummy
        num_terms = 100     # dummy  
        num_zero_classes = 100  # dummy
        num_rels = 100      # dummy
        device = 'cpu'
        
        model = DGELModel(num_interpros, num_terms, num_zero_classes, num_rels, device)
        print("Model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
            
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def get_real_model_parameters():
    """Get the real parameters needed for model creation"""
    print("\n=== GETTING REAL MODEL PARAMETERS ===")
    
    try:
        import pandas as pd
        
        # Load the actual parameter files
        terms_file = "/data/shared/tools/deepgozero/data/bp/terms.pkl"
        interpros_file = "/data/shared/tools/deepgozero/data/bp/interpros.pkl"
        
        if os.path.exists(terms_file):
            terms_df = pd.read_pickle(terms_file)
            print(f"Terms shape: {terms_df.shape}")
            print(f"Terms columns: {terms_df.columns.tolist()}")
            
        if os.path.exists(interpros_file):
            interpros_df = pd.read_pickle(interpros_file)
            print(f"InterPros shape: {interpros_df.shape}")
            print(f"InterPros columns: {interpros_df.columns.tolist()}")
            
    except Exception as e:
        print(f"Error loading parameter files: {e}")

def main():
    """Run comprehensive DeepGOZero diagnostics"""
    print("=== DEEPGOZERO COMPREHENSIVE DIAGNOSTIC ===")
    
    check_pytorch_version()
    check_model_files()
    
    # Try loading models
    model = try_load_with_different_methods()
    
    # Check architecture
    arch_model = check_deepgozero_architecture()
    
    # Get real parameters
    get_real_model_parameters()
    
    if model is None:
        print("\n❌ CONCLUSION: All model loading methods failed")
        print("RECOMMENDATIONS:")
        print("1. Re-download DeepGOZero model files")
        print("2. Check PyTorch version compatibility")
        print("3. Try a different DeepGOZero installation")
        print("4. Use alternative method (InterPro mapping)")
    else:
        print("\n✅ CONCLUSION: Found working model!")
        print("Try using this model for predictions")

if __name__ == "__main__":
    main()