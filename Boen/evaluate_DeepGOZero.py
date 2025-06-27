#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd  # This is already here, good
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

logging.basicConfig(level=logging.INFO)

ont = 'bp'  # Changed from 'mf' to 'bp'

@ck.command()
@ck.option(
    '--data-root', '-dr', default='/data/shared/tools/deepgozero/data/',  # FULL PATH
    help='Data root')
@ck.option(
    '--ont', '-ont', default='bp',  # Changed from 'mf' to 'bp'
    help='Subontology')
@ck.option(
    '--data-file', '-df', default='/data/summer2020/Boen/deepgozero_pipeline_output/prediction_input_bp_fixed.pkl',  # FULL PATH
    help='Pandas pkl file with proteins and their interpo annotations')
@ck.option(
    '--device', '-d', default='cpu',  # Changed from 'cuda:1' to 'cpu'
    help='Device')
@ck.option(
    '--output-file', '-o', default='/data/summer2020/Boen/deepgozero_pipeline_output/deepgozero_predictions_bp.csv',
    help='Output CSV file for predictions')
def main(data_root, ont, data_file, device, output_file):
    import pandas as pd  # Add this import here
    terms_file = f'{data_root}/{ont}/terms.pkl'
    model_file = f'{data_root}/{ont}/deepgozero.th'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    # Load interpro data
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
    print(f"Zero-shot terms: {len(zero_terms)}")
    
    # Create model with correct dimensions
    net = DGELModel(len(iprs_dict), len(terms), len(zero_classes), len(rels_dict), device).to(device)
    
    # FIXED MODEL LOADING - Handle missing BatchNorm running statistics
    print('Loading the model with missing key handling...')
    try:
        # Load the state dict
        state_dict = th.load(model_file, map_location=device)
        
        # Load with strict=False to allow missing keys
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will use default values): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
            
        # Initialize missing BatchNorm running statistics manually
        for name, module in net.named_modules():
            if isinstance(module, th.nn.BatchNorm1d):
                if not hasattr(module, 'running_mean') or module.running_mean is None:
                    module.running_mean = th.zeros(module.num_features)
                if not hasattr(module, 'running_var') or module.running_var is None:
                    module.running_var = th.ones(module.num_features)
                    
        print("Model loaded successfully with missing key handling!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    net.eval()

    zero_terms_dict = {v: k for k, v in enumerate(zero_terms)}
    data = get_data(df, iprs_dict, zero_terms_dict)
    
    print(f"Data tensor shape: {data[0].shape}")
    print(f"Processing {data[0].shape[0]} proteins...")
    
    # Check how many proteins have features
    proteins_with_features = (data[0].sum(dim=1) > 0).sum().item()
    print(f"Proteins with InterPro features: {proteins_with_features}/{data[0].shape[0]}")
    
    batch_size = 1000
    data_loader = FastTensorDataLoader(*data, batch_size=batch_size, shuffle=False)

    go_data = th.zeros(len(zero_terms), dtype=th.long).to(device)
    for i, term in enumerate(zero_terms):
        go_data[i] = zero_classes[term]
        
    scores = np.zeros((data[0].shape[0], len(zero_terms)), dtype=np.float32)
    
    print("Running predictions...")
    for i, batch_data in enumerate(data_loader):
        batch_data, _ = batch_data
        with th.no_grad():  # Add no_grad for efficiency
            zero_score = net.predict_zero(
                batch_data.to(device), go_data).cpu().detach().numpy()
        scores[i * batch_size: (i + 1) * batch_size] = zero_score
        
        if i % 10 == 0:
            print(f"Processed batch {i+1}/{len(data_loader)}")
    
    print("Generating predictions...")
    prediction_count = 0
    predictions = []  # Store predictions for CSV output
    
    for i, row in enumerate(df.itertuples()):
        for j, go_id in enumerate(zero_terms):
            if scores[i, j] >= 0.01:  # Threshold for output
                print(row.proteins, go_id, scores[i, j])
                predictions.append({
                    'Protein_ID': row.proteins,
                    'GO_Term': go_id,
                    'Score': float(scores[i, j])
                })
                prediction_count += 1
                
    print(f"Total predictions generated: {prediction_count}")
    
    # Save predictions to CSV
    if predictions:
        pred_df = pd.DataFrame(predictions)  # pd should work now
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
        print(f"Summary: {len(pred_df)} predictions for {pred_df['Protein_ID'].nunique()} proteins")
    else:
        print("WARNING: No predictions generated!")
        
    return prediction_count
                
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0
    rmax = 0
    patience = 0
    precs = []
    recs = []
    for t in range(0, 101):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        tp_ind = tp > 0
        tp = tp[tp_ind]
        fp = fp[tp_ind]
        fn = fn[tp_ind]
        if len(tp) == 0:
            continue
        p = np.mean(tp / (tp + fp))
        r = np.sum(tp / (tp + fn)) / len(tp_ind)
        precs.append(p)
        recs.append(r)
        f = 2 * p * r / (p + r)
        if fmax <= f:
            fmax = f
    return fmax, precs, recs


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    
    print(f"Creating data matrix: {len(df)} proteins x {len(iprs_dict)} InterPro features")
    
    proteins_with_interpros = 0
    for i, row in enumerate(df.itertuples()):
        has_interpro = False
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
                has_interpro = True
        if has_interpro:
            proteins_with_interpros += 1
            
        # Handle labels (prop_annotations might be empty for prediction)
        if hasattr(row, 'prop_annotations') and row.prop_annotations:
            for go_id in row.prop_annotations:
                if go_id in terms_dict:
                    g_id = terms_dict[go_id]
                    labels[i, g_id] = 1
                    
    print(f"Proteins with matching InterPro domains: {proteins_with_interpros}/{len(df)}")
    return data, labels

if __name__ == '__main__':
    main()