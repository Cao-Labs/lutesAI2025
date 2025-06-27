#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import sys
import torch as th

# ADD DEEPGOZERO TO PYTHON PATH
sys.path.insert(0, '/data/shared/tools/deepgozero')

from collections import Counter
import logging
import json

from sklearn.metrics import roc_curve, auc, matthews_corrcoef

# Import from DeepGOZero (now in path)
try:
    from aminoacids import MAXLEN, to_ngrams
    from utils import get_goplus_defs, Ontology, NAMESPACES
    from deepgozero import DGELModel, load_normal_forms
    from torch_utils import FastTensorDataLoader
    print("✅ All DeepGOZero imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure DeepGOZero is installed at /data/shared/tools/deepgozero")
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)  # Reduced verbosity

@ck.command()
@ck.option(
    '--data-root', '-dr', default='/data/shared/tools/deepgozero/data/',
    help='Data root')
@ck.option(
    '--ont', '-ont', default='bp',
    help='Subontology')
@ck.option(
    '--data-file', '-df', default='/data/summer2020/Boen/deepgozero_pipeline_output/prediction_input_bp_fixed.pkl',
    help='Pandas pkl file with proteins and their interpo annotations')
@ck.option(
    '--device', '-d', default='cpu',
    help='Device')
@ck.option(
    '--output-file', '-o', default='/data/summer2020/Boen/deepgozero_pipeline_output/deepgozero_predictions_bp.csv',
    help='Output CSV file for predictions')
@ck.option(
    '--threshold', '-t', default=0.0, type=float,
    help='Minimum score threshold for predictions (default: 0.0)')
@ck.option(
    '--model-name', '-m', default='deepgozero_zero.th',
    help='Model file name (try: deepgozero_zero.th, deepgozero_zero_10.th)')
@ck.option(
    '--verbose', '-v', is_flag=True,
    help='Enable verbose output')
def main(data_root, ont, data_file, device, output_file, threshold, model_name, verbose):
    print(f"🚀 Starting DeepGOZero prediction for {ont} ontology")
    print(f"📊 Using threshold: {threshold} (0 = all predictions)")
    print(f"💾 Output file: {output_file}")
    print("-" * 60)
    
    # Load required files
    terms_file = f'{data_root}/{ont}/terms.pkl'
    model_file = f'{data_root}/{ont}/deepgozero.th'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    print("📁 Loading data files...")
    df = pd.read_pickle(data_file)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    nf1, nf2, nf3, nf4, rels_dict, zero_classes = load_normal_forms(
        f'{data_root}/go.norm', terms_dict)

    defins = get_goplus_defs(f'{data_root}/definitions_go.txt')
    zero_terms = [term for term in zero_classes if term in defins and go.get_namespace(term) == NAMESPACES[ont]]
    
    print(f"📈 Data summary:")
    print(f"   • Proteins: {len(df)}")
    print(f"   • InterPro features: {len(iprs_dict)}")
    print(f"   • Zero-shot GO terms: {len(zero_terms)}")
    
    # Create and load model
    print("🧠 Loading model...")
    net = DGELModel(len(iprs_dict), len(terms), len(zero_classes), len(rels_dict), device).to(device)
    
    # Try strict loading first
    try:
        state_dict = th.load(model_file, map_location=device)
        net.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully (strict mode)")
        
    except Exception as e:
        print(f"⚠️  Strict loading failed: {str(e)[:100]}...")
        print("🔄 Trying flexible loading...")
        
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} (may affect performance)")
            if verbose:
                print(f"   Missing: {missing_keys}")
                
        # Initialize missing BatchNorm statistics
        for name, module in net.named_modules():
            if isinstance(module, th.nn.BatchNorm1d):
                if not hasattr(module, 'running_mean') or module.running_mean is None:
                    module.running_mean = th.zeros(module.num_features)
                if not hasattr(module, 'running_var') or module.running_var is None:
                    module.running_var = th.ones(module.num_features)
                    
        print("✅ Model loaded with flexible mode")
    
    net.eval()

    # Prepare data
    print("🔧 Processing input data...")
    zero_terms_dict = {v: k for k, v in enumerate(zero_terms)}
    data = get_data(df, iprs_dict, zero_terms_dict, verbose)
    
    # Setup data loader
    batch_size = 1000
    data_loader = FastTensorDataLoader(*data, batch_size=batch_size, shuffle=False)

    go_data = th.zeros(len(zero_terms), dtype=th.long).to(device)
    for i, term in enumerate(zero_terms):
        go_data[i] = zero_classes[term]
        
    scores = np.zeros((data[0].shape[0], len(zero_terms)), dtype=np.float32)
    
    # Run predictions
    print("🔮 Running predictions...")
    total_batches = len(data_loader)
    
    for i, batch_data in enumerate(data_loader):
        batch_data, _ = batch_data
        with th.no_grad():
            zero_score = net.predict_zero(
                batch_data.to(device), go_data).cpu().detach().numpy()
        scores[i * batch_size: (i + 1) * batch_size] = zero_score
        
        # Progress update every 20% or every 50 batches, whichever is less frequent
        progress_interval = max(1, min(50, total_batches // 5))
        if (i + 1) % progress_interval == 0 or i == total_batches - 1:
            progress = (i + 1) / total_batches * 100
            print(f"   Progress: {progress:.1f}% ({i+1}/{total_batches} batches)")
    
    # Generate predictions
    print("📝 Generating final predictions...")
    predictions = []
    prediction_count = 0
    proteins_with_predictions = 0
    
    for i, row in enumerate(df.itertuples()):
        protein_predictions = 0
        for j, go_id in enumerate(zero_terms):
            if scores[i, j] >= threshold:
                predictions.append({
                    'Protein_ID': row.proteins,
                    'GO_Term': go_id,
                    'Score': float(scores[i, j])
                })
                prediction_count += 1
                protein_predictions += 1
                
        if protein_predictions > 0:
            proteins_with_predictions += 1
            
        # Show progress for large datasets
        if len(df) > 1000 and (i + 1) % (len(df) // 10) == 0:
            progress = (i + 1) / len(df) * 100
            print(f"   Processing proteins: {progress:.0f}%")
    
    # Save results
    print("💾 Saving results...")
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values(['Protein_ID', 'Score'], ascending=[True, False])
        pred_df.to_csv(output_file, index=False)
        
        print("✅ COMPLETED!")
        print("-" * 60)
        print("📊 RESULTS SUMMARY:")
        print(f"   • Total predictions: {len(pred_df):,}")
        print(f"   • Proteins with predictions: {proteins_with_predictions:,}/{len(df):,} ({proteins_with_predictions/len(df)*100:.1f}%)")
        print(f"   • Average predictions per protein: {len(pred_df)/proteins_with_predictions:.1f}")
        print(f"   • Score range: {pred_df['Score'].min():.3f} - {pred_df['Score'].max():.3f}")
        print(f"   • Median score: {pred_df['Score'].median():.3f}")
        print(f"   • Output saved to: {output_file}")
        
        # Show top predictions sample
        if verbose:
            print("\n🔍 Sample top predictions:")
            sample = pred_df.head(10)
            for _, row in sample.iterrows():
                print(f"   {row['Protein_ID']} -> {row['GO_Term']} (score: {row['Score']:.3f})")
        
    else:
        print("⚠️  WARNING: No predictions generated!")
        print(f"   All scores below threshold {threshold}")
        print(f"   Try lowering the threshold or check your data")
        
    return prediction_count
                
def get_data(df, iprs_dict, terms_dict, verbose=False):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    
    if verbose:
        print(f"   Creating data matrix: {len(df)} proteins × {len(iprs_dict)} InterPro features")
    
    proteins_with_interpros = 0
    total_interpro_matches = 0
    
    for i, row in enumerate(df.itertuples()):
        has_interpro = False
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
                has_interpro = True
                total_interpro_matches += 1
                
        if has_interpro:
            proteins_with_interpros += 1
            
        # Handle labels (prop_annotations might be empty for prediction)
        if hasattr(row, 'prop_annotations') and row.prop_annotations:
            for go_id in row.prop_annotations:
                if go_id in terms_dict:
                    g_id = terms_dict[go_id]
                    labels[i, g_id] = 1
    
    feature_coverage = proteins_with_interpros / len(df) * 100
    avg_features = total_interpro_matches / len(df)
    
    print(f"   • Proteins with InterPro features: {proteins_with_interpros:,}/{len(df):,} ({feature_coverage:.1f}%)")
    print(f"   • Average InterPro domains per protein: {avg_features:.1f}")
    
    if feature_coverage < 50:
        print("   ⚠️  Low feature coverage - predictions may be limited")
    
    return data, labels

if __name__ == '__main__':
    main()