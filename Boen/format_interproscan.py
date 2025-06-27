#!/usr/bin/env python

import pandas as pd
import click as ck
from collections import defaultdict
from Bio import SeqIO
import os

@ck.command()
@ck.option('--interpro-tsv', '-i', required=True, help='InterProScan TSV output file')
@ck.option('--fasta-file', '-f', required=True, help='Original FASTA file with protein sequences')
@ck.option('--output-pkl', '-o', required=True, help='Output pickle file for DeepGOZero')
@ck.option('--include-all-proteins', is_flag=True, help='Include proteins with no InterPro hits (empty interpros list)')
def main(interpro_tsv, fasta_file, output_pkl, include_all_proteins):
    """
    Convert InterProScan TSV output to DeepGOZero input format.
    
    InterProScan TSV columns:
    0: Protein Accession
    1: Sequence MD5 digest
    2: Sequence Length
    3: Analysis
    4: Signature Accession
    5: Signature Description
    6: Start location
    7: Stop location
    8: Score
    9: Status
    10: Date
    11: InterPro annotations - accession
    12: InterPro annotations - description
    13: GO annotations
    14: Pathways annotations
    """
    
    print("Loading InterProScan results...")
    
    # Read InterProScan TSV file
    try:
        interpro_df = pd.read_csv(interpro_tsv, sep='\t', header=None, 
                                 names=['protein_id', 'md5', 'length', 'analysis', 
                                       'signature_acc', 'signature_desc', 'start', 'stop',
                                       'score', 'status', 'date', 'interpro_acc', 
                                       'interpro_desc', 'go_terms', 'pathways'])
    except Exception as e:
        print(f"Error reading InterProScan file: {e}")
        return
    
    print(f"Loaded {len(interpro_df)} InterProScan records")
    
    # Load original FASTA to get all protein IDs
    print("Loading original FASTA file...")
    fasta_proteins = set()
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            fasta_proteins.add(record.id)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return
    
    print(f"Found {len(fasta_proteins)} proteins in FASTA file")
    
    # Group data by protein
    protein_data = defaultdict(lambda: {'interpros': set(), 'go_terms': set()})
    
    # Process InterProScan results
    for _, row in interpro_df.iterrows():
        protein_id = row['protein_id']
        
        # Add InterPro accessions
        if pd.notna(row['interpro_acc']):
            protein_data[protein_id]['interpros'].add(row['interpro_acc'])
        
        # Add GO terms
        if pd.notna(row['go_terms']):
            # GO terms can be pipe-separated in InterProScan output
            go_terms = str(row['go_terms']).split('|')
            for go_term in go_terms:
                go_term = go_term.strip()
                if go_term and go_term.startswith('GO:'):
                    protein_data[protein_id]['go_terms'].add(go_term)
    
    # Create final DataFrame
    proteins_with_interpro = set(protein_data.keys())
    
    if include_all_proteins:
        # Include all proteins from FASTA file
        all_proteins = fasta_proteins
        print(f"Including all {len(all_proteins)} proteins from FASTA file")
    else:
        # Only include proteins that have InterPro hits
        all_proteins = proteins_with_interpro
        print(f"Including only {len(all_proteins)} proteins with InterPro annotations")
    
    # Build the final dataset
    final_data = []
    proteins_with_interpro_count = 0
    proteins_with_go_count = 0
    
    for protein_id in all_proteins:
        interpros = list(protein_data[protein_id]['interpros']) if protein_id in protein_data else []
        go_terms = list(protein_data[protein_id]['go_terms']) if protein_id in protein_data else []
        
        if interpros:
            proteins_with_interpro_count += 1
        if go_terms:
            proteins_with_go_count += 1
        
        final_data.append({
            'proteins': protein_id,
            'interpros': interpros,
            'prop_annotations': go_terms  # Using GO terms as annotations
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(final_data)
    
    # Print statistics
    print("\n=== CONVERSION STATISTICS ===")
    print(f"Total proteins processed: {len(result_df)}")
    print(f"Proteins with InterPro annotations: {proteins_with_interpro_count}")
    print(f"Proteins with GO annotations: {proteins_with_go_count}")
    print(f"Proteins with no annotations: {len(result_df) - proteins_with_interpro_count}")
    
    # Show unique InterPro and GO term counts
    all_interpros = set()
    all_go_terms = set()
    for _, row in result_df.iterrows():
        all_interpros.update(row['interpros'])
        all_go_terms.update(row['prop_annotations'])
    
    print(f"Unique InterPro terms: {len(all_interpros)}")
    print(f"Unique GO terms: {len(all_go_terms)}")
    
    # Show sample data
    print("\n=== SAMPLE DATA ===")
    for i in range(min(3, len(result_df))):
        row = result_df.iloc[i]
        print(f"Protein: {row['proteins']}")
        print(f"  InterPros ({len(row['interpros'])}): {row['interpros'][:5]}{'...' if len(row['interpros']) > 5 else ''}")
        print(f"  GO terms ({len(row['prop_annotations'])}): {row['prop_annotations'][:3]}{'...' if len(row['prop_annotations']) > 3 else ''}")
        print()
    
    # Save to pickle
    print(f"Saving to {output_pkl}...")
    result_df.to_pickle(output_pkl)
    print("Conversion complete!")
    
    # Save summary statistics
    stats_file = output_pkl.replace('.pkl', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("DeepGOZero Input File Statistics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total proteins: {len(result_df)}\n")
        f.write(f"Proteins with InterPro annotations: {proteins_with_interpro_count}\n")
        f.write(f"Proteins with GO annotations: {proteins_with_go_count}\n")
        f.write(f"Unique InterPro terms: {len(all_interpros)}\n")
        f.write(f"Unique GO terms: {len(all_go_terms)}\n")
        f.write(f"Average InterPro terms per protein: {sum(len(row['interpros']) for _, row in result_df.iterrows()) / len(result_df):.2f}\n")
        f.write(f"Average GO terms per protein: {sum(len(row['prop_annotations']) for _, row in result_df.iterrows()) / len(result_df):.2f}\n")
    
    print(f"Statistics saved to {stats_file}")

if __name__ == '__main__':
    main()