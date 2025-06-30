#!/usr/bin/env python

import click as ck
import pandas as pd
import numpy as np
import torch as th
import time
import os
from pathlib import Path
import pickle
from Bio import SeqIO
from collections import defaultdict
import logging
import json
from typing import Dict, List, Tuple, Optional

# Assuming these modules exist in your environment
from utils import Ontology, get_goplus_defs, NAMESPACES
from deepgozero import DGELModel, load_normal_forms
from torch_utils import FastTensorDataLoader

logging.basicConfig(level=logging.INFO)

class DeepGOZeroBenchmark:
    """
    Benchmarking suite for DeepGOZero model on large protein datasets
    """
    
    def __init__(self, data_root: str, ont: str = 'mf', device: str = 'cuda:0'):
        self.data_root = data_root
        self.ont = ont
        self.device = device
        self.go = None
        self.model = None
        self.terms_dict = None
        self.iprs_dict = None
        self.zero_classes = None
        
    def load_model_and_data(self):
        """Load the trained model and necessary data structures"""
        print(f"Loading model and data for ontology: {self.ont}")
        
        # Load Gene Ontology
        self.go = Ontology(f'{self.data_root}/go.obo', with_rels=True)
        
        # Load terms and InterPro dictionaries
        terms_file = f'{self.data_root}/{self.ont}/terms.pkl'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        self.terms_dict = {v: i for i, v in enumerate(terms)}
        
        ipr_file = f'{self.data_root}/{self.ont}/interpros.pkl'
        ipr_df = pd.read_pickle(ipr_file)
        iprs = ipr_df['interpros'].values
        self.iprs_dict = {v: k for k, v in enumerate(iprs)}
        
        # Load normal forms for zero-shot prediction
        nf1, nf2, nf3, nf4, rels_dict, self.zero_classes = load_normal_forms(
            f'{self.data_root}/go.norm', self.terms_dict)
        
        # Load the trained model
        model_file = f'{self.data_root}/{self.ont}/deepgozero.th'
        self.model = DGELModel(
            len(self.iprs_dict), 
            len(self.terms_dict), 
            len(self.zero_classes), 
            len(rels_dict), 
            self.device
        ).to(self.device)
        
        print(f"Loading model weights from: {model_file}")
        self.model.load_state_dict(th.load(model_file, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"- Terms: {len(self.terms_dict)}")
        print(f"- InterPro domains: {len(self.iprs_dict)}")
        print(f"- Zero-shot classes: {len(self.zero_classes)}")
        
    def parse_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Parse FASTA file and return protein sequences"""
        print(f"Parsing FASTA file: {fasta_file}")
        sequences = {}
        
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences[record.id] = str(record.seq)
                
        print(f"Loaded {len(sequences)} protein sequences")
        return sequences
        
    def load_interpro_annotations(self, interpro_file: str) -> pd.DataFrame:
        """
        Load pre-computed InterPro annotations from pickle file
        """
        print(f"Loading InterPro annotations from: {interpro_file}")
        
        try:
            df = pd.read_pickle(interpro_file)
            print(f"Loaded annotations for {len(df)} proteins")
            
            # Validate required columns
            required_cols = ['proteins']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"WARNING: Missing required columns: {missing_cols}")
            
            # Check for InterPro annotations column
            if 'interpros' not in df.columns:
                print("WARNING: 'interpros' column not found. Looking for alternatives...")
                # Try common alternative column names
                alt_names = ['interpro', 'interpro_domains', 'ipr', 'domains']
                for alt in alt_names:
                    if alt in df.columns:
                        df['interpros'] = df[alt]
                        print(f"Using '{alt}' column as InterPro annotations")
                        break
                else:
                    print("ERROR: No InterPro annotation column found!")
                    print(f"Available columns: {list(df.columns)}")
                    raise ValueError("InterPro annotations not found")
            
            # Ensure GO annotations column exists (optional)
            if 'prop_annotations' not in df.columns:
                if 'go_annotations' in df.columns:
                    df['prop_annotations'] = df['go_annotations']
                else:
                    df['prop_annotations'] = [[] for _ in range(len(df))]
                    print("No GO annotations found - using empty lists")
            
            # Print summary statistics
            total_proteins = len(df)
            proteins_with_interpro = sum(1 for interpros in df['interpros'] if interpros and len(interpros) > 0)
            total_interpro_annotations = sum(len(interpros) if interpros else 0 for interpros in df['interpros'])
            
            print(f"Summary:")
            print(f"  Total proteins: {total_proteins}")
            print(f"  Proteins with InterPro annotations: {proteins_with_interpro} ({proteins_with_interpro/total_proteins:.1%})")
            print(f"  Total InterPro annotations: {total_interpro_annotations}")
            print(f"  Average annotations per protein: {total_interpro_annotations/total_proteins:.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise
        
    def prepare_input_data(self, df: pd.DataFrame) -> Tuple[th.Tensor, th.Tensor]:
        """Convert DataFrame to model input format"""
        print("Preparing input data...")
        
        data = th.zeros((len(df), len(self.iprs_dict)), dtype=th.float32)
        labels = th.zeros((len(df), len(self.terms_dict)), dtype=th.float32)
        
        for i, row in enumerate(df.itertuples()):
            # Set InterPro features
            for ipr in row.interpros:
                if ipr in self.iprs_dict:
                    data[i, self.iprs_dict[ipr]] = 1
                    
            # Set known annotations (if available)
            if hasattr(row, 'prop_annotations') and row.prop_annotations:
                for go_id in row.prop_annotations:
                    if go_id in self.terms_dict:
                        labels[i, self.terms_dict[go_id]] = 1
                        
        return data, labels
        
    def run_predictions(self, data: th.Tensor, batch_size: int = 1000) -> np.ndarray:
        """Run model predictions on input data"""
        print(f"Running predictions on {data.shape[0]} proteins...")
        
        n_samples = data.shape[0]
        predictions = np.zeros((n_samples, len(self.terms_dict)), dtype=np.float32)
        
        # Create data loader
        dummy_labels = th.zeros((n_samples, len(self.terms_dict)), dtype=th.float32)
        data_loader = FastTensorDataLoader(data, dummy_labels, batch_size=batch_size, shuffle=False)
        
        with th.no_grad():
            start_time = time.time()
            for i, (batch_data, _) in enumerate(data_loader):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_samples)
                
                batch_data = batch_data.to(self.device)
                logits = self.model(batch_data)
                predictions[batch_start:batch_end] = logits.cpu().numpy()
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {batch_end}/{n_samples} proteins ({elapsed:.2f}s)")
                    
        return predictions
        
    def run_zero_shot_predictions(self, data: th.Tensor, batch_size: int = 1000) -> np.ndarray:
        """Run zero-shot predictions for unseen GO terms"""
        print("Running zero-shot predictions...")
        
        # Get zero-shot terms for this ontology
        defins = get_goplus_defs(f'{self.data_root}/definitions_go.txt')
        zero_terms = [term for term in self.zero_classes 
                     if term in defins and self.go.get_namespace(term) == NAMESPACES[self.ont]]
        
        if not zero_terms:
            print("No zero-shot terms found for this ontology")
            return np.array([])
            
        print(f"Found {len(zero_terms)} zero-shot terms")
        
        # Prepare zero-shot data
        go_data = th.zeros(len(zero_terms), dtype=th.long).to(self.device)
        for i, term in enumerate(zero_terms):
            go_data[i] = self.zero_classes[term]
            
        n_samples = data.shape[0]
        zero_predictions = np.zeros((n_samples, len(zero_terms)), dtype=np.float32)
        
        # Create data loader
        dummy_labels = th.zeros((n_samples, 1), dtype=th.float32)
        data_loader = FastTensorDataLoader(data, dummy_labels, batch_size=batch_size, shuffle=False)
        
        with th.no_grad():
            for i, (batch_data, _) in enumerate(data_loader):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_samples)
                
                batch_data = batch_data.to(self.device)
                zero_scores = self.model.predict_zero(batch_data, go_data)
                zero_predictions[batch_start:batch_end] = zero_scores.cpu().numpy()
                
        return zero_predictions, zero_terms
        
    def validate_interpro_annotations(self, df: pd.DataFrame) -> Dict:
        """
        Validate InterPro annotations against the model's known domains
        """
        print("Validating InterPro annotations against model's training set...")
        
        # Get all unique InterPro domains from annotations
        all_interpros = set()
        proteins_with_annotations = 0
        total_annotations = 0
        
        for _, row in df.iterrows():
            interpros = row.get('interpros', [])
            if interpros and len(interpros) > 0:
                proteins_with_annotations += 1
                total_annotations += len(interpros)
                all_interpros.update(interpros)
        
        # Check coverage against model's known InterPro domains
        known_interpros = set(self.iprs_dict.keys())
        covered_interpros = all_interpros.intersection(known_interpros)
        unknown_interpros = all_interpros - known_interpros
        
        coverage_rate = len(covered_interpros) / len(all_interpros) if all_interpros else 0
        
        stats = {
            'total_proteins': len(df),
            'proteins_with_annotations': proteins_with_annotations,
            'annotation_coverage': proteins_with_annotations / len(df),
            'total_interpro_annotations': total_annotations,
            'avg_annotations_per_protein': total_annotations / len(df),
            'unique_interpros_in_data': len(all_interpros),
            'interpros_known_to_model': len(covered_interpros),
            'interpros_unknown_to_model': len(unknown_interpros),
            'domain_coverage_rate': coverage_rate,
            'known_interpros_sample': list(covered_interpros)[:10],
            'unknown_interpros_sample': list(unknown_interpros)[:10]
        }
        
        print(f"Validation Results:")
        print(f"  Total proteins: {stats['total_proteins']:,}")
        print(f"  Proteins with InterPro annotations: {stats['proteins_with_annotations']:,} ({stats['annotation_coverage']:.1%})")
        print(f"  Total InterPro annotations: {stats['total_interpro_annotations']:,}")
        print(f"  Average annotations per protein: {stats['avg_annotations_per_protein']:.2f}")
        print(f"  Unique InterPro domains in data: {stats['unique_interpros_in_data']:,}")
        print(f"  Domains known to model: {stats['interpros_known_to_model']:,}")
        print(f"  Unknown domains: {stats['interpros_unknown_to_model']:,}")
        print(f"  Domain coverage rate: {stats['domain_coverage_rate']:.1%}")
        
        if unknown_interpros:
            print(f"  Sample unknown domains: {list(unknown_interpros)[:5]}")
        
        return stats

    def propagate_predictions(self, predictions: np.ndarray, protein_ids: List[str], min_score: float = 0.01) -> List[Dict]:
        """Apply ontology structure to propagate predictions"""
        print(f"Propagating predictions using ontology structure (min_score={min_score})...")
        
        results = []
        significant_predictions = 0
        
        for i, (prot_id, scores) in enumerate(zip(protein_ids, predictions)):
            prop_annots = {}
            
            # Get predictions above threshold and propagate
            for go_id, j in self.terms_dict.items():
                score = scores[j]
                if score >= min_score:
                    # Add direct prediction
                    prop_annots[go_id] = score
                    
                    # Propagate to ancestors
                    try:
                        for sup_go in self.go.get_anchestors(go_id):
                            if sup_go in prop_annots:
                                prop_annots[sup_go] = max(prop_annots[sup_go], score)
                            else:
                                prop_annots[sup_go] = score
                    except:
                        # If ancestor propagation fails, continue with direct prediction
                        pass
                        
            if prop_annots:
                significant_predictions += 1
                        
            results.append({
                'protein_id': prot_id,
                'predictions': prop_annots
            })
            
        print(f"Propagation complete: {significant_predictions}/{len(protein_ids)} proteins with predictions ≥ {min_score}")
        return results
        
    def benchmark_speed(self, n_proteins: int = 1000, batch_sizes: List[int] = [32, 64, 128, 256, 512, 1024]):
        """Benchmark prediction speed with different batch sizes"""
        print(f"Benchmarking prediction speed with {n_proteins} proteins...")
        
        # Create dummy data
        dummy_data = th.randn((n_proteins, len(self.iprs_dict)), dtype=th.float32)
        
        results = {}
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            start_time = time.time()
            _ = self.run_predictions(dummy_data, batch_size=batch_size)
            elapsed_time = time.time() - start_time
            
            proteins_per_second = n_proteins / elapsed_time
            results[batch_size] = {
                'elapsed_time': elapsed_time,
                'proteins_per_second': proteins_per_second
            }
            
            print(f"  Time: {elapsed_time:.2f}s, Speed: {proteins_per_second:.1f} proteins/sec")
            
        return results
        
    def save_results(self, results: List[Dict], output_file: str):
        """Save prediction results"""
        print(f"Saving results to: {output_file}")
        
        # Convert to DataFrame for easier analysis
        data = []
        for result in results:
            prot_id = result['protein_id']
            for go_term, score in result['predictions'].items():
                data.append({
                    'protein_id': prot_id,
                    'go_term': go_term,
                    'score': score,
                    'go_name': self.go.get_term(go_term).get('name', 'Unknown') if self.go.has_term(go_term) else 'Unknown'
                })
                
        df = pd.DataFrame(data)
        
        # Save in multiple formats
        df.to_csv(output_file.replace('.pkl', '.csv'), index=False)
        df.to_pickle(output_file)
        
        # Save summary statistics
        summary = {
            'total_proteins': len(results),
            'total_predictions': len(df),
            'avg_predictions_per_protein': len(df) / len(results),
            'score_distribution': {
                'min': float(df['score'].min()),
                'max': float(df['score'].max()),
                'mean': float(df['score'].mean()),
                'std': float(df['score'].std())
            }
        }
        
        with open(output_file.replace('.pkl', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Results saved successfully!")
        print(f"Summary: {len(results)} proteins, {len(df)} predictions")


@ck.command()
@ck.option('--interpro-file', '-ipr', required=True, help='Input pickle file with InterPro annotations')
@ck.option('--data-root', '-dr', default='data', help='Data root directory')
@ck.option('--ont', '-ont', default='mf', help='Ontology (mf, bp, cc)')
@ck.option('--output-dir', '-o', default='benchmark_results', help='Output directory')
@ck.option('--batch-size', '-bs', default=1000, help='Batch size for predictions')
@ck.option('--device', '-d', default='cuda:0', help='Device (cuda:0, cpu)')
@ck.option('--benchmark-speed', is_flag=True, help='Run speed benchmarking')
@ck.option('--zero-shot', is_flag=True, help='Include zero-shot predictions')
@ck.option('--min-score', '-ms', default=0.01, help='Minimum prediction score threshold')
@ck.option('--validate-only', is_flag=True, help='Only validate annotations without running predictions')
def main(interpro_file, data_root, ont, output_dir, batch_size, device, benchmark_speed, zero_shot, min_score, validate_only):
    """
    Benchmark DeepGOZero model on a large protein dataset with pre-computed InterPro annotations
    """
    print("=" * 60)
    print("DeepGOZero Benchmarking Suite")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize benchmark suite
    benchmark = DeepGOZeroBenchmark(data_root, ont, device)
    benchmark.load_model_and_data()
    
    # Load InterPro annotations
    df = benchmark.load_interpro_annotations(interpro_file)
    
    # Validate annotations against model's known domains
    print("\n" + "=" * 40)
    print("VALIDATING ANNOTATIONS")
    print("=" * 40)
    
    validation_stats = benchmark.validate_interpro_annotations(df)
    
    # Save validation results
    with open(f'{output_dir}/validation_stats_{ont}.json', 'w') as f:
        json.dump(validation_stats, f, indent=2)
    
    if validate_only:
        print("Validation completed. Exiting without running predictions.")
        return
    
    if benchmark_speed:
        print("\n" + "=" * 40)
        print("SPEED BENCHMARKING")
        print("=" * 40)
        speed_results = benchmark.benchmark_speed()
        
        # Save speed results
        with open(f'{output_dir}/speed_benchmark_{ont}.json', 'w') as f:
            json.dump(speed_results, f, indent=2)
        return
    
    # Prepare input data
    data, labels = benchmark.prepare_input_data(df)
    
    print(f"\nInput data prepared:")
    print(f"  Data shape: {data.shape}")
    print(f"  Average InterPro features per protein: {data.sum(dim=1).mean():.2f}")
    print(f"  Proteins with known GO annotations: {labels.sum(dim=0).sum().item()}")
    
    # Run standard predictions
    print("\n" + "=" * 40)
    print("RUNNING STANDARD PREDICTIONS")
    print("=" * 40)
    
    predictions = benchmark.run_predictions(data, batch_size=batch_size)
    
    # Run zero-shot predictions if requested
    zero_results = []
    if zero_shot:
        print("\n" + "=" * 40)
        print("RUNNING ZERO-SHOT PREDICTIONS")
        print("=" * 40)
        
        zero_predictions, zero_terms = benchmark.run_zero_shot_predictions(data, batch_size=batch_size)
        
        if len(zero_predictions) > 0:
            # Process zero-shot results
            for i, prot_id in enumerate(df['proteins']):
                zero_preds = {}
                for j, term in enumerate(zero_terms):
                    if zero_predictions[i, j] >= min_score:
                        zero_preds[term] = float(zero_predictions[i, j])
                        
                if zero_preds:
                    zero_results.append({
                        'protein_id': prot_id,
                        'predictions': zero_preds
                    })
                    
            if zero_results:
                benchmark.save_results(zero_results, f'{output_dir}/zero_shot_predictions_{ont}.pkl')
                print(f"Zero-shot predictions saved for {len(zero_results)} proteins")
    
    # Process and save standard predictions
    print("\n" + "=" * 40)
    print("PROCESSING RESULTS")
    print("=" * 40)
    
    protein_ids = df['proteins'].tolist()
    results = benchmark.propagate_predictions(predictions, protein_ids, min_score=min_score)
    
    # Filter results to only include proteins with significant predictions
    filtered_results = [result for result in results if result['predictions']]
    
    if filtered_results:
        benchmark.save_results(filtered_results, f'{output_dir}/predictions_{ont}.pkl')
        
        print(f"\nBenchmarking completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Standard predictions: {len(filtered_results)}/{len(df)} proteins")
        if zero_results:
            print(f"Zero-shot predictions: {len(zero_results)}/{len(df)} proteins")
        
        # Print some example predictions
        print(f"\nExample predictions (first 5 proteins):")
        for i, result in enumerate(filtered_results[:5]):
            prot_id = result['protein_id']
            n_preds = len(result['predictions'])
            top_pred = max(result['predictions'].items(), key=lambda x: x[1])
            print(f"  {prot_id}: {n_preds} predictions, top: {top_pred[0]} ({top_pred[1]:.3f})")
            
    else:
        print("\nWARNING: No significant predictions found!")
        print("This might be due to:")
        print("1. InterPro domains not matching model's training set")
        print("2. Very low prediction scores (try lowering --min-score)")
        print("3. Model/data compatibility issues")
        print(f"\nValidation showed {validation_stats['coverage']:.1%} domain coverage")
        
        # Save empty results file for debugging
        with open(f'{output_dir}/no_predictions_{ont}.json', 'w') as f:
            json.dump({
                'message': 'No significant predictions found',
                'validation_stats': validation_stats,
                'min_score_threshold': min_score
            }, f, indent=2)


if __name__ == '__main__':
    main()